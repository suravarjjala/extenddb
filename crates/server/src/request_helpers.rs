// Copyright 2026 ExtendDB contributors
// SPDX-License-Identifier: Apache-2.0

//! Request parsing and authorization helpers for the `DynamoDB` wire protocol.

use axum::http::HeaderMap;
use extenddb_core::error::DynamoDbError;
use extenddb_storage::error::StorageError;
use serde_json::Value;

use crate::AppState;
use crate::authorization;

/// Extract operation name from X-Amz-Target header.
/// Accepts both `DynamoDB_20120810` and `DynamoDBStreams_20120810` wire-format prefixes.
pub(crate) fn extract_operation(headers: &HeaderMap) -> Result<String, DynamoDbError> {
    let target = headers
        .get("x-amz-target")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            // S-7: Real DynamoDB returns MissingAuthenticationToken only when auth
            // headers are also absent. When auth headers are present but X-Amz-Target
            // is missing, it returns UnknownOperationException.
            if headers.contains_key("authorization") {
                DynamoDbError::UnknownOperationException(String::new())
            } else {
                DynamoDbError::MissingAuthenticationToken("Missing Authentication Token".to_owned())
            }
        })?;

    target
        .strip_prefix("DynamoDB_20120810.")
        .or_else(|| target.strip_prefix("DynamoDBStreams_20120810."))
        .map(std::borrow::ToOwned::to_owned)
        .ok_or_else(|| DynamoDbError::UnknownOperationException(String::new()))
}

/// Extract the table name from a `DynamoDB` request body.
///
/// Most operations use `TableName`. Batch and transact operations embed table
/// names in nested structures — returns `None` for those; the caller maps
/// `None` to `*` via `build_resource_arn`.
pub(crate) fn extract_table_name(input: &Value) -> Option<String> {
    input
        .get("TableName")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
}

/// What the auth path learned about the request's table, for the engine.
#[derive(Debug, Default)]
pub(crate) struct KeyInfoPrefetch {
    /// The table's key info, when the lookup found an ACTIVE table.
    pub info: Option<extenddb_core::types::TableKeyInfo>,
    /// The lookup ran and the name has no ACTIVE table, so the engine need
    /// not look it up again. Never set when the lookup failed with a
    /// storage error: the engine then looks the name up itself and reports
    /// that error (retryable), rather than a not-found the SDKs do not retry.
    pub known_missing: bool,
}

impl KeyInfoPrefetch {
    /// Classify the auth path's lookup (`None`: no lookup ran).
    pub(crate) fn from_lookup(
        lookup: Option<Result<Option<extenddb_core::types::TableKeyInfo>, StorageError>>,
    ) -> Self {
        match lookup {
            Some(Ok(Some(info))) => Self {
                info: Some(info),
                known_missing: false,
            },
            Some(Ok(None)) => Self {
                info: None,
                known_missing: true,
            },
            Some(Err(error)) => {
                tracing::debug!(
                    "key info lookup on the auth path failed ({error}); the engine looks \
                     the table up again"
                );
                Self::default()
            }
            None => Self::default(),
        }
    }
}

/// Evaluate IAM policies for an authenticated identity.
///
/// Returns what the auth path learned about the table of a single-table
/// item-level operation (P118 optimization #2): its `TableKeyInfo`, or that
/// it has no ACTIVE table. The caller passes this into `OperationContext` to
/// avoid a redundant catalog roundtrip in the engine layer.
///
/// All authorization data is fetched via `state.authz_cache`, which sits on
/// top of the underlying `AuthorizationStore` and serves cached, pre-parsed
/// `PolicyDocument`s.
pub(crate) async fn authorize_request(
    state: &AppState,
    identity: &extenddb_auth::AuthIdentity,
    input: &Value,
    operation: &str,
    account_id: &str,
    source_ip: Option<&str>,
    user_agent: Option<&str>,
) -> Result<KeyInfoPrefetch, DynamoDbError> {
    let table_name = extract_table_name(input);

    // P118: Fetch table_key_info for item-level operations via the SWR cache.
    // The result is used for LeadingKeys extraction here AND returned to the
    // caller to avoid a redundant fetch in the engine layer.
    let lookup = match (table_name.as_deref(), operation) {
        (Some(tn), op) if extenddb_engine::KEY_INFO_PREFETCH_OPERATIONS.contains(&op) => Some(
            state
                .table_key_info_cache
                .get_optional(account_id, tn)
                .await,
        ),
        _ => None,
    };
    let prefetch = KeyInfoPrefetch::from_lookup(lookup);

    let pk_attr = prefetch
        .info
        .as_ref()
        .map(|ki| ki.key_schema[0].attribute_name.clone());

    let context = RequestContext {
        region: &state.region,
        secure_transport: state.tls_enabled,
        source_ip,
        user_agent,
    };
    evaluate_request_policies(
        state.authz_cache.as_ref(),
        state.dev_mode,
        &context,
        identity,
        input,
        operation,
        account_id,
        pk_attr.as_deref(),
    )
    .await?;

    Ok(prefetch)
}

/// The request-level condition keys every evaluation of a request carries
/// (`aws:RequestedRegion`, `aws:SecureTransport`, `aws:SourceIp`,
/// `aws:UserAgent`), whichever branch of [`evaluate_request_policies`] it
/// takes.
pub(crate) struct RequestContext<'a> {
    /// The server's region: `aws:RequestedRegion`, and the region of every
    /// resource ARN built for the request.
    pub region: &'a str,
    pub secure_transport: bool,
    pub source_ip: Option<&'a str>,
    pub user_agent: Option<&'a str>,
}

impl RequestContext<'_> {
    /// `RequestParams` with the request-level keys set and nothing else.
    fn params(&self) -> extenddb_auth::policy::context::RequestParams {
        extenddb_auth::policy::context::RequestParams {
            region: Some(self.region.to_owned()),
            secure_transport: self.secure_transport,
            source_ip: self.source_ip.map(ToOwned::to_owned),
            user_agent: self.user_agent.map(ToOwned::to_owned),
            ..Default::default()
        }
    }
}

/// The IAM decision for one request: every (action, resource) pair the
/// operation authorizes on must be allowed.
///
/// Split from [`authorize_request`] so it can be exercised without an
/// `AppState`; `pk_attr` is the table's partition-key attribute for item
/// operations (for `dynamodb:LeadingKeys`), `None` otherwise.
#[allow(clippy::too_many_arguments)]
async fn evaluate_request_policies(
    cache: &crate::authz_cache::CachedAuthzStore,
    dev_mode: bool,
    context: &RequestContext<'_>,
    identity: &extenddb_auth::AuthIdentity,
    input: &Value,
    operation: &str,
    account_id: &str,
    pk_attr: Option<&str>,
) -> Result<(), DynamoDbError> {
    // Developer mode opens authorization for the authenticated caller. SigV4
    // verification already ran upstream, so the request is authenticated; only
    // the IAM policy decision is skipped.
    //
    // This check is deliberately positioned ahead of EVERY authorization branch
    // below, not merely the generic one. It previously sat after the batch and
    // transaction branch, which returns early, so those four operations were
    // still evaluated against IAM and denied in a dev-mode build while single
    // item operations passed. Any future operation-specific branch must be added
    // below this point, or it will reintroduce that defect.
    if dev_mode {
        return Ok(());
    }
    let region = context.region;

    // Batch and transaction operations address multiple tables in nested request
    // structures with no top-level TableName. DynamoDB authorizes each of them
    // per table, against that table's specific ARN, using the IAM action the
    // operation maps to (BatchGetItem/BatchWriteItem are their own actions;
    // TransactGetItems decomposes to GetItem; TransactWriteItems decomposes to
    // the per-sub-op action). Authorizing the whole request against a single
    // `table/*` wildcard, as the generic path below does, lets an explicit
    // Deny on one table be bypassed. Evaluate every (action, table) pair and
    // reject the entire request if any is denied (all-or-nothing), matching
    // DynamoDB. Verified against the AWS IAM Service Authorization Reference.
    if let Some(targets) = batch_transact_authz_targets(operation, input) {
        for (action_op, table) in targets {
            let resource_arn = build_resource_arn(region, account_id, Some(&table));
            authorization::check_authorization(
                cache,
                identity,
                &action_op,
                &resource_arn,
                false,
                context.params(),
            )
            .await?;
        }
        return Ok(());
    }

    // Backup and restore operations name their resources in members other
    // than `TableName` (`BackupArn`, `TargetTableName`, `SourceTableName`), so
    // the generic path below would authorize them against `table/*` and a
    // policy scoped to a backup or a target table could never match. Evaluate
    // every (action, resource) pair the service authorizes on, all of which
    // must allow; `RestoreTableFromBackup` authorizes the target table under
    // the item actions the restore depends on, not under its own name.
    if let Some(targets) = backup_authz_targets(operation, input, region, account_id) {
        for (action, resource_arn) in targets? {
            authorization::check_authorization(
                cache,
                identity,
                &action,
                &resource_arn,
                false,
                context.params(),
            )
            .await?;
        }
        return Ok(());
    }

    let table_name = extract_table_name(input);
    let resource_arn = build_resource_arn(region, account_id, table_name.as_deref());

    let params = extenddb_auth::policy::context::RequestParams {
        leading_keys: extract_leading_keys(input, operation, pk_attr),
        attributes: extract_attributes(input),
        select: input
            .get("Select")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        return_values: input
            .get("ReturnValues")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        return_consumed_capacity: input
            .get("ReturnConsumedCapacity")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        ..context.params()
    };
    authorization::check_authorization(
        cache,
        identity,
        operation,
        &resource_arn,
        operation == "Scan",
        params,
    )
    .await
}

/// Extract partition key values from the request body for `dynamodb:LeadingKeys`.
///
/// For item-level operations (`GetItem`, `PutItem`, `DeleteItem`, `UpdateItem`),
/// extracts the partition key value from the `Key` or `Item` using the table's
/// key schema. For `Query`, the leading key comes from `KeyConditionExpression`
/// values, but extracting that requires expression parsing — deferred to the
/// engine layer.
/// Returns `None` for table-level and batch/transact operations, or when
/// `pk_attr` is not available.
fn extract_leading_keys(
    input: &Value,
    operation: &str,
    pk_attr: Option<&str>,
) -> Option<Vec<String>> {
    let pk_attr = pk_attr?;
    match operation {
        "GetItem" | "DeleteItem" | "UpdateItem" => extract_pk_value(input.get("Key")?, pk_attr),
        "PutItem" => extract_pk_value(input.get("Item")?, pk_attr),
        _ => None,
    }
}

/// Extract the partition key value from a `DynamoDB` key/item map using the
/// known PK attribute name.
///
/// `DynamoDB` keys are `{"attrName": {"S": "value"}}`. We extract the typed
/// value of the partition key attribute as a string.
fn extract_pk_value(map: &Value, pk_attr: &str) -> Option<Vec<String>> {
    let obj = map.as_object()?;
    let type_val = obj.get(pk_attr)?;
    let type_obj = type_val.as_object()?;
    let (_, val) = type_obj.iter().next()?;
    let s = val.as_str().unwrap_or_default();
    Some(vec![s.to_owned()])
}

/// Extract attribute names from the request for `dynamodb:Attributes`.
///
/// Collects attribute names from `ProjectionExpression` (comma-separated list
/// of top-level names). Resolves `ExpressionAttributeNames` placeholders
/// (e.g. `#n` → `name`) when present.
/// Returns `None` when no projection is specified.
pub(crate) fn extract_attributes(input: &Value) -> Option<Vec<String>> {
    let proj = input.get("ProjectionExpression")?.as_str()?;
    let ean = input
        .get("ExpressionAttributeNames")
        .and_then(|v| v.as_object());
    let names: Vec<String> = proj
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            ean.and_then(|m| m.get(s))
                .and_then(|v| v.as_str())
                .unwrap_or(s)
                .to_owned()
        })
        .collect();
    if names.is_empty() { None } else { Some(names) }
}

/// IAM `(action, table)` pairs to authorize for a batch/transact operation.
///
/// Returns `None` for non-batch/transact operations (the caller uses the
/// single-table path). Actions follow the AWS IAM Service Authorization
/// Reference: `BatchGetItem`/`BatchWriteItem` authorize under their own action
/// per table; `TransactGetItems` authorizes as `GetItem` per item's table;
/// `TransactWriteItems` authorizes per sub-op (`Put`→`PutItem`,
/// `Delete`→`DeleteItem`, `Update`→`UpdateItem`, `ConditionCheck`→
/// `ConditionCheckItem`). Pairs are de-duplicated. Table names are the bare
/// operation-string form (no `dynamodb:` prefix), matching `check_authorization`
/// which prepends `dynamodb:`.
fn batch_transact_authz_targets(operation: &str, input: &Value) -> Option<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut push = |action: &str, table: &str| {
        let pair = (action.to_owned(), table.to_owned());
        if !out.contains(&pair) {
            out.push(pair);
        }
    };
    match operation {
        "BatchGetItem" | "BatchWriteItem" => {
            // RequestItems is a map keyed by table name.
            let items = input.get("RequestItems")?.as_object()?;
            for table in items.keys() {
                push(operation, table);
            }
        }
        "TransactGetItems" => {
            let items = input.get("TransactItems")?.as_array()?;
            for entry in items {
                if let Some(t) = entry
                    .get("Get")
                    .and_then(|g| g.get("TableName"))
                    .and_then(Value::as_str)
                {
                    push("GetItem", t);
                }
            }
        }
        "TransactWriteItems" => {
            let items = input.get("TransactItems")?.as_array()?;
            for entry in items {
                for (sub, action) in [
                    ("Put", "PutItem"),
                    ("Delete", "DeleteItem"),
                    ("Update", "UpdateItem"),
                    ("ConditionCheck", "ConditionCheckItem"),
                ] {
                    if let Some(t) = entry
                        .get(sub)
                        .and_then(|s| s.get("TableName"))
                        .and_then(Value::as_str)
                    {
                        push(action, t);
                    }
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

/// `(action, resource ARN)` pairs to authorize for a backup or restore
/// operation.
///
/// Returns `None` for any other operation (the caller uses the generic
/// `TableName` path). The resource types follow the AWS Service Authorization
/// Reference for Amazon `DynamoDB`
/// (<https://docs.aws.amazon.com/service-authorization/latest/reference/list_dynamodb.html>,
/// also published as
/// <https://servicereference.us-east-1.amazonaws.com/v1/dynamodb/dynamodb.json>;
/// both read 2026-10-15); the `ListBackups` resource and the
/// `RestoreTableFromBackup` actions follow what the service was measured to
/// evaluate (2026-10-20), which the reference does not spell out:
///
/// | Operation | Evaluated here (every pair must allow) |
/// |---|---|
/// | `CreateBackup` | a `TableName` given as the table's ARN: `CreateBackup` on the `table/<name>` the ARN names (this function's own arm). A plain `TableName`: generic path, `CreateBackup` on `table/<TableName>` |
/// | `DescribeContinuousBackups` | generic path: on `table/<TableName>` |
/// | `UpdateContinuousBackups` | generic path: on `table/<TableName>` |
/// | `DescribeBackup` | `DescribeBackup` on the `BackupArn` (both spellings, below) |
/// | `DeleteBackup` | `DeleteBackup` on the `BackupArn` (both spellings) |
/// | `RestoreTableFromBackup` | `RestoreTableFromBackup` on the `BackupArn` (both spellings); then `Scan`, `Query`, `UpdateItem`, `PutItem`, `GetItem`, `DeleteItem`, and `BatchWriteItem` on `table/<TargetTableName>`. `RestoreTableFromBackup` is never evaluated on the target |
/// | `ListBackups` | `ListBackups` on `table/*/backup/*`, or on `table/<TableName>/backup/*` when `TableName` filters the list |
/// | `RestoreTableToPointInTime` | on `table/<SourceTableName>` (the source, per the developer guide's backup IAM page); `table/*` for `SourceTableArn` |
///
/// The `backup` ARN format is
/// `arn:${Partition}:dynamodb:${Region}:${Account}:table/${TableName}/backup/${BackupName}`.
///
/// A well-formed `BackupArn` is evaluated on every ARN the engine may act
/// on. The engine trims surrounding ASCII whitespace, resolves the ARN
/// rebuilt in this deployment's region from the trimmed parts first (the
/// canonical spelling), and, when that backup is not found and the trimmed
/// ARN as written differs from it (another region of the caller's account),
/// resolves the ARN as written. Both spellings are therefore evaluated, the
/// canonical one first, and both must allow: a Deny on either stops the
/// request whichever the engine would have resolved. An ARN that differs
/// from its canonical spelling only by surrounding whitespace is one
/// spelling and is evaluated once, trimmed.
///
/// A `BackupArn` that is missing, not a string, or not a backup ARN by
/// [`extenddb_storage::util::parse_backup_arn`] is evaluated against
/// `table/*`, the resource every one of these operations was evaluated against
/// before this check existed, so no caller is allowed more than before. Such a
/// request names no backup; the engine then answers `ValidationException`.
/// A missing `TargetTableName` likewise contributes `table/*` (under each of
/// the dependent item actions). A `ListBackups` whose `TableName` is not a
/// string is evaluated as unfiltered, `table/*/backup/*`.
///
/// A backup ARN in a partition other than `aws` is treated the same way,
/// before the account is compared, matching the engine's order.
///
/// A well-formed `BackupArn` naming another account is refused here with the
/// engine's own `AccessDeniedException` and no policy evaluation. `DynamoDB`
/// backups take no resource-based policy, so an identity policy cannot grant
/// access to another account's backup and the service denies it whatever
/// the caller's policies say. Evaluating it would also look up the other
/// account's resource tags for `aws:ResourceTag` conditions, which a caller
/// could use to probe them.
///
/// Pairs are returned in evaluation order; the first denied pair names the
/// action and resource in the `AccessDeniedException`.
fn backup_authz_targets(
    operation: &str,
    input: &Value,
    region: &str,
    account_id: &str,
) -> Option<Result<Vec<(String, String)>, DynamoDbError>> {
    let wildcard = || build_resource_arn(region, account_id, None);
    let str_member = |name: &str| input.get(name).and_then(Value::as_str);
    let own = |resource: String| (operation.to_owned(), resource);
    // The backup resources to evaluate under `operation`: one for a missing,
    // malformed, or other-partition ARN (`table/*`); the canonical spelling,
    // then the trimmed ARN as written when it differs, for a well-formed one.
    let backups = || -> Result<Vec<String>, DynamoDbError> {
        let Some(arn) = str_member("BackupArn") else {
            return Ok(vec![wildcard()]);
        };
        match extenddb_storage::util::parse_backup_arn(arn) {
            // Another partition names no backup of this deployment; the
            // engine answers it with ValidationException before it looks at
            // the account, so it is evaluated as a malformed ARN here too: a
            // caller allowed on `table/*` gets that ValidationException, one
            // allowed only on backup ARNs gets AccessDenied.
            None => Ok(vec![wildcard()]),
            Some(parts) if parts.partition != extenddb_storage::util::ARN_PARTITION => {
                Ok(vec![wildcard()])
            }
            Some(parts) if parts.account_id != account_id => {
                tracing::warn!(
                    action = format!("dynamodb:{operation}"),
                    resource = arn,
                    "Authorization denied: backup ARN names another account"
                );
                Err(DynamoDbError::AccessDeniedException(
                    "Access is denied".to_owned(),
                ))
            }
            // The ARNs the engine may resolve: the canonical spelling
            // (rebuilt in this deployment's region from the trimmed parts),
            // which it tries first, then the trimmed ARN as written, which
            // it falls back to when the canonical one is not found and the
            // two differ. Both must allow.
            Some(parts) => {
                let canonical = extenddb_storage::util::backup_arn(
                    region,
                    parts.account_id,
                    parts.table_name,
                    parts.backup_id,
                );
                let mut out = vec![canonical];
                if parts.arn != out[0] {
                    out.push(parts.arn.to_owned());
                }
                Ok(out)
            }
        }
    };
    let targets = match operation {
        // `CreateBackup` accepts `TableName` as the table's ARN. The generic
        // path would build `table/arn:aws:...` from it; evaluate the table
        // the ARN names instead. A plain name stays on the generic path. An
        // ARN the engine refuses (ValidationException for a malformed one,
        // AccessDenied for another account) is evaluated as `table/*`.
        "CreateBackup" => match str_member("TableName") {
            Some(value) if value.starts_with("arn:") => Ok(vec![own(build_resource_arn(
                region,
                account_id,
                table_name_of_table_arn(value, account_id),
            ))]),
            _ => return None,
        },
        "DescribeBackup" | "DeleteBackup" => backups().map(|b| b.into_iter().map(own).collect()),
        "RestoreTableFromBackup" => backups().map(|b| {
            let target = build_resource_arn(region, account_id, str_member("TargetTableName"));
            let mut out: Vec<(String, String)> =
                Vec::with_capacity(b.len() + RESTORE_TARGET_ACTIONS.len());
            out.extend(b.into_iter().map(own));
            out.extend(
                RESTORE_TARGET_ACTIONS
                    .iter()
                    .map(|action| ((*action).to_owned(), target.clone())),
            );
            out
        }),
        "ListBackups" => Ok(vec![own(build_backup_wildcard_arn(
            region,
            account_id,
            str_member("TableName"),
        ))]),
        "RestoreTableToPointInTime" => {
            let mut out = Vec::new();
            if let Some(name) = str_member("SourceTableName") {
                out.push(own(build_resource_arn(region, account_id, Some(name))));
            }
            // ExtendDB does not implement point-in-time recovery and the
            // engine refuses the operation; a SourceTableArn is not resolved
            // here and keeps the pre-existing `table/*` evaluation.
            if out.is_empty() || input.get("SourceTableArn").is_some() {
                out.push(own(wildcard()));
            }
            Ok(out)
        }
        _ => return None,
    };
    Some(targets)
}

/// The actions `RestoreTableFromBackup` is authorized under on the target
/// table, in the order they are evaluated. The service requires each of them
/// (the restore reads and writes the target with them) and does not evaluate
/// `RestoreTableFromBackup` itself on the target. Measured against the
/// service on 2026-10-20.
const RESTORE_TARGET_ACTIONS: [&str; 7] = [
    "Scan",
    "Query",
    "UpdateItem",
    "PutItem",
    "GetItem",
    "DeleteItem",
    "BatchWriteItem",
];

/// The table name a `CreateBackup` `TableName` given as a table ARN names,
/// when the ARN is one the engine resolves (this partition, a digit-only
/// account equal to the caller's, `table/<name>` with no further path).
/// Any other value is `None`, which evaluates as `table/*` and leaves the
/// engine to answer with its own error.
fn table_name_of_table_arn<'a>(value: &'a str, account_id: &str) -> Option<&'a str> {
    let rest = value.strip_prefix("arn:")?;
    let fields: Vec<&str> = rest.split(':').collect();
    let [partition, "dynamodb", region, arn_account, resource] = fields.as_slice() else {
        return None;
    };
    if *partition != extenddb_storage::util::ARN_PARTITION
        || region.is_empty()
        || *arn_account != account_id
        || !arn_account.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let name = resource.strip_prefix("table/")?;
    if name.is_empty() || name.contains('/') {
        return None;
    }
    Some(name)
}

/// Build the backup-wildcard ARN `ListBackups` is authorized against:
/// `table/<TableName>/backup/*` for a filtered list, `table/*/backup/*`
/// otherwise. The `*` components are literal parts of the resource string;
/// the policy matcher's `*` crosses `/`, so a statement on `table/*` (or on
/// `table/<name>/*`) covers it, and one on `table/<name>` does not.
fn build_backup_wildcard_arn(region: &str, account_id: &str, table_name: Option<&str>) -> String {
    let table = table_name.unwrap_or("*");
    format!("arn:aws:dynamodb:{region}:{account_id}:table/{table}/backup/*")
}

/// Build a `DynamoDB` table ARN for authorization.
///
/// If no table name is available (e.g. `ListTables`, `DescribeEndpoints`),
/// uses `*` as the resource.
fn build_resource_arn(region: &str, account_id: &str, table_name: Option<&str>) -> String {
    match table_name {
        Some(name) => format!("arn:aws:dynamodb:{region}:{account_id}:table/{name}"),
        None => format!("arn:aws:dynamodb:{region}:{account_id}:table/*"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Only a lookup that found no ACTIVE table is known missing. A storage
    /// error is not: the engine looks the name up again and reports the
    /// error, instead of answering ResourceNotFoundException for a table that
    /// exists.
    #[test]
    fn only_a_not_found_lookup_is_known_missing() {
        use extenddb_core::types::TableKeyInfo;
        let found = KeyInfoPrefetch::from_lookup(Some(Ok(Some(TableKeyInfo::default()))));
        assert!(found.info.is_some() && !found.known_missing);
        let missing = KeyInfoPrefetch::from_lookup(Some(Ok(None)));
        assert!(missing.info.is_none() && missing.known_missing);
        for error in [
            StorageError::Connection("pool timed out".to_owned()),
            StorageError::Internal("catalog read failed".to_owned()),
        ] {
            let failed = KeyInfoPrefetch::from_lookup(Some(Err(error)));
            assert!(failed.info.is_none() && !failed.known_missing, "{failed:?}");
        }
        let not_looked_up = KeyInfoPrefetch::from_lookup(None);
        assert!(not_looked_up.info.is_none() && !not_looked_up.known_missing);
    }

    #[test]
    fn extract_attributes_resolves_expression_attribute_names() {
        let input = json!({
            "ProjectionExpression": "#n, #v",
            "ExpressionAttributeNames": {
                "#n": "name",
                "#v": "value"
            }
        });
        let result = extract_attributes(&input);
        assert_eq!(result, Some(vec!["name".to_owned(), "value".to_owned()]));
    }

    #[test]
    fn extract_attributes_mixed_placeholders_and_literals() {
        let input = json!({
            "ProjectionExpression": "#n, age",
            "ExpressionAttributeNames": {
                "#n": "name"
            }
        });
        let result = extract_attributes(&input);
        assert_eq!(result, Some(vec!["name".to_owned(), "age".to_owned()]));
    }

    #[test]
    fn extract_attributes_no_expression_attribute_names() {
        let input = json!({
            "ProjectionExpression": "name, age"
        });
        let result = extract_attributes(&input);
        assert_eq!(result, Some(vec!["name".to_owned(), "age".to_owned()]));
    }

    #[test]
    fn extract_attributes_no_projection() {
        let input = json!({"TableName": "test"});
        assert_eq!(extract_attributes(&input), None);
    }

    #[test]
    fn missing_target_no_auth_returns_missing_auth_token() {
        // S-7: No Authorization header + no X-Amz-Target → MissingAuthenticationToken
        let headers = HeaderMap::new();
        let err = extract_operation(&headers).unwrap_err();
        assert!(
            matches!(err, DynamoDbError::MissingAuthenticationToken(_)),
            "Expected MissingAuthenticationToken, got: {err:?}"
        );
    }

    #[test]
    fn missing_target_with_auth_returns_unknown_operation() {
        // S-7: Authorization header present but no X-Amz-Target → UnknownOperationException
        use axum::http::HeaderValue;
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("AWS4-HMAC-SHA256 Credential=AKID/20260415/us-east-1/dynamodb/aws4_request, SignedHeaders=host, Signature=abc"));
        let err = extract_operation(&headers).unwrap_err();
        assert!(
            matches!(err, DynamoDbError::UnknownOperationException(_)),
            "Expected UnknownOperationException, got: {err:?}"
        );
    }
}

/// Authorization of the backup and restore operations: the `(action,
/// resource)` pairs each one is evaluated against, and policy decisions over
/// the input space of `BackupArn` (own table, another table, another region,
/// surrounding whitespace, another account, malformed, missing),
/// `TargetTableName` (present, missing), `CreateBackup`'s `TableName` (a
/// name, a table ARN) and `ListBackups`'s `TableName` (present, missing, not
/// a string).
#[cfg(test)]
mod backup_authz_tests {
    use std::sync::{Arc, Mutex};

    use extenddb_auth::AuthIdentity;
    use extenddb_auth::policy::matcher::arn_match;
    use extenddb_storage::authorization_store::{AuthorizationStore, SessionData};
    use extenddb_storage::management_store::OpResult;
    use futures::future::BoxFuture;
    use serde_json::{Value, json};

    use super::{
        RESTORE_TARGET_ACTIONS, RequestContext, backup_authz_targets, evaluate_request_policies,
    };
    use crate::authz_cache::{AuthzCacheConfig, CachedAuthzStore};
    use extenddb_core::error::DynamoDbError;

    const REGION: &str = "us-east-1";
    const ACCT: &str = "123456789012";
    const OTHER_ACCT: &str = "210987654321";
    const USER: &str = "alice";
    const ROLE: &str = "restorer";
    const SESSION: &str = "s1";
    const RESTORE: &str = "RestoreTableFromBackup";

    fn table(name: &str) -> String {
        format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/{name}")
    }
    fn backup(table_name: &str, id: &str) -> String {
        format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/{table_name}/backup/{id}")
    }
    /// The resource `ListBackups` is evaluated against: `table/<t>/backup/*`,
    /// or `table/*/backup/*` without a filter.
    fn backups_of(table_name: Option<&str>) -> String {
        backup(table_name.unwrap_or("*"), "*")
    }
    fn wildcard() -> String {
        table("*")
    }
    const A: &str = "01700000000000-0000000a";
    const B: &str = "01700000000000-0000000b";

    /// `(action, resource)` pairs, the shape `backup_authz_targets` returns.
    type Pairs = Vec<(String, String)>;

    fn pair(action: &str, resource: &str) -> (String, String) {
        (action.to_owned(), resource.to_owned())
    }
    /// Every pair under one action.
    fn under(action: &str, resources: &[String]) -> Pairs {
        resources.iter().map(|r| pair(action, r)).collect()
    }
    /// The pairs a restore of `backup_arn` into `target` is evaluated on:
    /// the restore action on the backup, then each dependent item action on
    /// the target.
    fn restore_pairs(backup_arn: &str, target: &str) -> Pairs {
        restore_pairs_on(std::slice::from_ref(&backup_arn.to_owned()), target)
    }
    /// The same for a backup evaluated under more than one spelling: the
    /// restore action on each spelling in order, then the item actions.
    fn restore_pairs_on(backup_arns: &[String], target: &str) -> Pairs {
        let mut out = under(RESTORE, backup_arns);
        out.extend(RESTORE_TARGET_ACTIONS.iter().map(|a| pair(a, target)));
        out
    }
    /// A backup ARN of this account in another region.
    fn backup_elsewhere(table_name: &str, id: &str) -> String {
        format!("arn:aws:dynamodb:eu-west-1:{ACCT}:table/{table_name}/backup/{id}")
    }

    /// Policies for one user and one role; records the ARNs whose resource
    /// tags were fetched.
    #[derive(Default)]
    struct Store {
        policies: Vec<String>,
        /// Tags the store holds per resource ARN.
        resource_tags: std::collections::HashMap<String, Vec<(String, String)>>,
        resource_tag_lookups: Mutex<Vec<String>>,
    }

    impl AuthorizationStore for Store {
        fn fetch_user_policies(&self, _: &str, _: &str) -> BoxFuture<'_, OpResult<Vec<String>>> {
            let v = self.policies.clone();
            Box::pin(async move { Ok(v) })
        }
        fn fetch_user_group_policies(
            &self,
            _: &str,
            _: &str,
        ) -> BoxFuture<'_, OpResult<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn fetch_user_boundary(&self, _: &str, _: &str) -> BoxFuture<'_, OpResult<Option<String>>> {
            Box::pin(async { Ok(None) })
        }
        fn fetch_role_policies(&self, _: &str, _: &str) -> BoxFuture<'_, OpResult<Vec<String>>> {
            let v = self.policies.clone();
            Box::pin(async move { Ok(v) })
        }
        fn fetch_role_boundary(&self, _: &str, _: &str) -> BoxFuture<'_, OpResult<Option<String>>> {
            Box::pin(async { Ok(None) })
        }
        fn fetch_session_data(
            &self,
            _: &str,
            _: &str,
            _: &str,
        ) -> BoxFuture<'_, OpResult<Option<SessionData>>> {
            Box::pin(async { Ok(None) })
        }
        fn fetch_user_tags(
            &self,
            _: &str,
            _: &str,
        ) -> BoxFuture<'_, OpResult<Vec<(String, String)>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn fetch_role_tags(
            &self,
            _: &str,
            _: &str,
        ) -> BoxFuture<'_, OpResult<Vec<(String, String)>>> {
            Box::pin(async { Ok(Vec::new()) })
        }
        fn fetch_resource_tags(&self, arn: &str) -> BoxFuture<'_, OpResult<Vec<(String, String)>>> {
            self.resource_tag_lookups
                .lock()
                .unwrap()
                .push(arn.to_owned());
            let tags = self.resource_tags.get(arn).cloned().unwrap_or_default();
            Box::pin(async move { Ok(tags) })
        }
        fn fetch_user_created_at(
            &self,
            _: &str,
            _: &str,
        ) -> BoxFuture<'_, OpResult<Option<String>>> {
            Box::pin(async { Ok(None) })
        }
        fn fetch_role_created_at(
            &self,
            _: &str,
            _: &str,
        ) -> BoxFuture<'_, OpResult<Option<String>>> {
            Box::pin(async { Ok(None) })
        }
    }

    /// One statement: `(effect, actions, resources)`.
    type Stmt<'a> = (&'a str, &'a [&'a str], &'a [&'a str]);

    fn policy(statements: &[Stmt<'_>]) -> String {
        let st: Vec<Value> = statements
            .iter()
            .map(|(effect, actions, resources)| {
                json!({"Effect": effect, "Action": actions, "Resource": resources})
            })
            .collect();
        json!({"Version": "2012-10-17", "Statement": st}).to_string()
    }

    fn harness(statements: &[Stmt<'_>]) -> (Arc<Store>, CachedAuthzStore) {
        let store = Arc::new(Store {
            policies: if statements.is_empty() {
                Vec::new()
            } else {
                vec![policy(statements)]
            },
            ..Store::default()
        });
        let cache = CachedAuthzStore::pass_through(store.clone(), AuthzCacheConfig::default());
        (store, cache)
    }

    fn user() -> AuthIdentity {
        AuthIdentity::User {
            account_id: ACCT.to_owned(),
            user_name: USER.to_owned(),
        }
    }

    fn role() -> AuthIdentity {
        AuthIdentity::RoleSession {
            account_id: ACCT.to_owned(),
            role_name: ROLE.to_owned(),
            session_name: SESSION.to_owned(),
        }
    }

    async fn decide_as(
        cache: &CachedAuthzStore,
        identity: &AuthIdentity,
        dev_mode: bool,
        op: &str,
        input: &Value,
    ) -> Result<(), DynamoDbError> {
        decide_from(cache, identity, dev_mode, None, op, input).await
    }

    /// `decide_as` for a request whose connection peer is `source_ip`.
    async fn decide_from(
        cache: &CachedAuthzStore,
        identity: &AuthIdentity,
        dev_mode: bool,
        source_ip: Option<&str>,
        op: &str,
        input: &Value,
    ) -> Result<(), DynamoDbError> {
        let context = RequestContext {
            region: REGION,
            secure_transport: true,
            source_ip,
            user_agent: Some("aws-sdk-test"),
        };
        evaluate_request_policies(cache, dev_mode, &context, identity, input, op, ACCT, None).await
    }

    async fn decide(
        cache: &CachedAuthzStore,
        op: &str,
        input: &Value,
    ) -> Result<(), DynamoDbError> {
        decide_as(cache, &user(), false, op, input).await
    }

    fn assert_denied(r: &Result<(), DynamoDbError>, what: &str) {
        assert!(
            matches!(r, Err(DynamoDbError::AccessDeniedException(_))),
            "{what}: expected AccessDeniedException, got {r:?}"
        );
    }

    /// Denied, and the message names `action` on `resource` (the first pair
    /// that failed).
    fn assert_denied_on(r: &Result<(), DynamoDbError>, action: &str, resource: &str, what: &str) {
        match r {
            Err(DynamoDbError::AccessDeniedException(m)) => assert!(
                m.contains(&format!(
                    "to perform: dynamodb:{action} on resource: {resource}"
                )),
                "{what}: denied, but not on dynamodb:{action} / {resource}: {m}"
            ),
            other => panic!("{what}: expected AccessDeniedException, got {other:?}"),
        }
    }

    fn targets(op: &str, input: &Value) -> Pairs {
        backup_authz_targets(op, input, REGION, ACCT)
            .unwrap_or_else(|| panic!("{op} must take the backup path"))
            .unwrap_or_else(|e| panic!("{op}: unexpected error {e:?}"))
    }

    fn restore(arn: &str, target: &str) -> Value {
        json!({"BackupArn": arn, "TargetTableName": target})
    }

    /// Malformed `BackupArn` values: each is refused by `parse_backup_arn`.
    fn malformed_arns() -> Vec<String> {
        vec![
            String::new(),
            "garbage".to_owned(),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/backup/"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/backup/{A}/x"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/stream/{A}"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table//backup/{A}"),
            format!("arn:aws:dynamodb:{REGION}::table/Music/backup/{A}"),
            format!("arn:aws:s3:{REGION}:{ACCT}:table/Music/backup/{A}"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/backup/{A}:extra"),
            format!("dynamodb:{REGION}:{ACCT}:table/Music/backup/{A}"),
        ]
    }

    /// A backup ARN in another partition, for this account or another, is
    /// evaluated against `table/*` (the engine refuses it with
    /// ValidationException); another account's ARN in this partition is still
    /// refused here.
    #[test]
    fn another_partitions_backup_arn_evaluates_table_wildcard() {
        for acct in [ACCT, "999999999999"] {
            let bad = format!("arn:aws-cn:dynamodb:{REGION}:{acct}:table/Music/backup/{A}");
            assert!(extenddb_storage::util::parse_backup_arn(&bad).is_some());
            for op in ["DescribeBackup", "DeleteBackup"] {
                assert_eq!(
                    targets(op, &json!({"BackupArn": bad})),
                    under(op, &[wildcard()]),
                    "{op} {bad}"
                );
            }
            assert_eq!(
                targets(RESTORE, &restore(&bad, "Restored")),
                restore_pairs(&wildcard(), &table("Restored")),
                "restore {bad}"
            );
        }
        let foreign = format!("arn:aws:dynamodb:{REGION}:999999999999:table/Music/backup/{A}");
        assert!(matches!(
            backup_authz_targets(
                "DescribeBackup",
                &json!({"BackupArn": foreign}),
                REGION,
                ACCT
            ),
            Some(Err(DynamoDbError::AccessDeniedException(_)))
        ));
    }

    // ── Pairs evaluated per operation ─────────────────────────────────

    #[test]
    fn describe_and_delete_backup_evaluate_the_backup_arn() {
        for op in ["DescribeBackup", "DeleteBackup"] {
            let arn = backup("Music", A);
            assert_eq!(targets(op, &json!({"BackupArn": arn})), under(op, &[arn]));
        }
    }

    /// The restore action on the backup ARN first, then each dependent item
    /// action on the target, in the order measured on `DynamoDB`; the
    /// restore action itself is never evaluated on the target.
    #[test]
    fn restore_evaluates_the_backup_arn_then_the_item_actions_on_the_target() {
        let arn = backup("Music", A);
        let got = targets(RESTORE, &restore(&arn, "Restored"));
        assert_eq!(got, restore_pairs(&arn, &table("Restored")));
        assert_eq!(got[0], pair(RESTORE, &arn));
        let on_target: Vec<&str> = got[1..].iter().map(|(a, _)| a.as_str()).collect();
        assert_eq!(
            on_target,
            [
                "Scan",
                "Query",
                "UpdateItem",
                "PutItem",
                "GetItem",
                "DeleteItem",
                "BatchWriteItem"
            ]
        );
        assert!(got[1..].iter().all(|(_, r)| *r == table("Restored")));
        assert!(
            !got.contains(&pair(RESTORE, &table("Restored"))),
            "RestoreTableFromBackup must not be evaluated on the target: {got:?}"
        );
        // The backup ARN is evaluated under the restore action only.
        assert_eq!(got.iter().filter(|(_, r)| *r == arn).count(), 1);
    }

    #[test]
    fn restore_without_target_name_evaluates_table_wildcard_for_the_target() {
        let arn = backup("Music", A);
        assert_eq!(
            targets(RESTORE, &json!({"BackupArn": arn})),
            restore_pairs(&arn, &wildcard())
        );
        // A non-string TargetTableName is treated as missing.
        assert_eq!(
            targets(RESTORE, &json!({"BackupArn": arn, "TargetTableName": 7})),
            restore_pairs(&arn, &wildcard())
        );
    }

    /// The table component of the ARN is part of the resource: a backup of
    /// Other is `table/Other/backup/<id>`, whatever table the request
    /// restores into. (The fixture is in the server's region, so the one
    /// spelling evaluated is the ARN as supplied.)
    #[test]
    fn backup_arn_for_another_table_name_is_evaluated_on_that_tables_backup() {
        let arn = backup("Other", A);
        for op in ["DescribeBackup", "DeleteBackup"] {
            assert_eq!(
                targets(op, &json!({"BackupArn": arn})),
                under(op, std::slice::from_ref(&arn))
            );
        }
        assert_eq!(
            targets(RESTORE, &restore(&arn, "Music")),
            restore_pairs(&arn, &table("Music"))
        );
    }

    /// The engine resolves an ARN in another region of the caller's account
    /// first as the ARN rebuilt in its own region and, when that backup is
    /// not found, as written; the policy is evaluated on both spellings, the
    /// rebuilt one first, and both must allow.
    #[test]
    fn backup_arn_in_another_region_is_evaluated_in_both_spellings() {
        let supplied = backup_elsewhere("Music", A);
        let resolved = backup("Music", A);
        assert_ne!(supplied, resolved);
        let both = [resolved.clone(), supplied.clone()];
        for op in ["DescribeBackup", "DeleteBackup"] {
            assert_eq!(
                targets(op, &json!({"BackupArn": supplied})),
                under(op, &both)
            );
        }
        assert_eq!(
            targets(RESTORE, &restore(&supplied, "Music")),
            restore_pairs_on(&both, &table("Music"))
        );
        // The backup is evaluated under the restore action only, once per
        // spelling; the item actions are on the target alone.
        let got = targets(RESTORE, &restore(&supplied, "Music"));
        assert_eq!(got.iter().filter(|(a, _)| a == RESTORE).count(), 2);
        assert!(got[2..].iter().all(|(_, r)| *r == table("Music")));
        // An ARN in the server's region is one spelling.
        for op in ["DescribeBackup", "DeleteBackup", RESTORE] {
            assert_eq!(
                targets(
                    op,
                    &json!({"BackupArn": resolved, "TargetTableName": "Music"})
                )
                .iter()
                .filter(|(_, r)| extenddb_storage::util::is_backup_resource_arn(r))
                .count(),
                1,
                "{op}"
            );
        }
    }

    /// The engine trims surrounding ASCII whitespace before it resolves the
    /// ARN (a cursor pasted with a trailing newline, for instance); the
    /// policy is evaluated on the trimmed ARN, and whitespace alone does not
    /// make a second spelling.
    #[test]
    fn backup_arn_with_surrounding_whitespace_is_evaluated_trimmed() {
        let resolved = backup("Music", A);
        for supplied in [
            format!("{resolved}\n"),
            format!("  {resolved}"),
            format!("\t{resolved} \r\n"),
        ] {
            assert_eq!(
                targets("DescribeBackup", &json!({"BackupArn": supplied})),
                under("DescribeBackup", std::slice::from_ref(&resolved)),
                "{supplied:?}"
            );
        }
        // Another region's ARN with whitespace: the rebuilt spelling, then
        // the trimmed one as written.
        let elsewhere = backup_elsewhere("Music", A);
        assert_eq!(
            targets(
                "DescribeBackup",
                &json!({"BackupArn": format!(" {elsewhere}\n")})
            ),
            under("DescribeBackup", &[resolved, elsewhere])
        );
    }

    /// `CreateBackup` with `TableName` given as the table's ARN is evaluated
    /// on the table the ARN names, in any region of the caller's account. A
    /// plain name is left to the generic path; an ARN the engine refuses is
    /// evaluated as `table/*`.
    #[test]
    fn create_backup_table_arn_is_evaluated_as_the_table_it_names() {
        let op = "CreateBackup";
        assert!(backup_authz_targets(op, &json!({"TableName": "Music"}), REGION, ACCT).is_none());
        assert!(backup_authz_targets(op, &json!({}), REGION, ACCT).is_none());
        for supplied in [
            table("Music"),
            format!("arn:aws:dynamodb:eu-west-1:{ACCT}:table/Music"),
        ] {
            assert_eq!(
                targets(op, &json!({"TableName": supplied, "BackupName": "b"})),
                under(op, &[table("Music")]),
                "{supplied}"
            );
        }
        for supplied in [
            format!("arn:aws:dynamodb:{REGION}:{OTHER_ACCT}:table/Music"),
            format!("arn:aws-cn:dynamodb:{REGION}:{ACCT}:table/Music"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/backup/{A}"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/"),
            format!("arn:aws:dynamodb:{REGION}:{ACCT}:index/Music"),
            format!("arn:aws:s3:::{ACCT}:table/Music"),
            format!("arn:aws:dynamodb:{REGION}:12ab:table/Music"),
            "arn:".to_owned(),
        ] {
            assert_eq!(
                targets(op, &json!({"TableName": supplied})),
                under(op, &[table("*")]),
                "{supplied}"
            );
        }
    }

    #[test]
    fn backup_arn_of_another_account_is_access_denied_without_evaluation() {
        let arn = format!("arn:aws:dynamodb:{REGION}:{OTHER_ACCT}:table/Music/backup/{A}");
        for (op, input) in [
            ("DescribeBackup", json!({"BackupArn": arn})),
            ("DeleteBackup", json!({"BackupArn": arn})),
            (RESTORE, restore(&arn, "Restored")),
            // The ARN account is checked before the target table name.
            (RESTORE, json!({"BackupArn": arn})),
        ] {
            let r = backup_authz_targets(op, &input, REGION, ACCT).expect("backup path");
            match r {
                Err(DynamoDbError::AccessDeniedException(m)) => assert_eq!(m, "Access is denied"),
                other => panic!("{op}: expected AccessDeniedException, got {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_backup_arn_evaluates_table_wildcard() {
        for bad in malformed_arns() {
            assert!(
                extenddb_storage::util::parse_backup_arn(&bad).is_none(),
                "fixture {bad:?} must be malformed"
            );
            for op in ["DescribeBackup", "DeleteBackup"] {
                assert_eq!(
                    targets(op, &json!({"BackupArn": bad})),
                    under(op, &[wildcard()]),
                    "{op} {bad:?}"
                );
            }
            assert_eq!(
                targets(RESTORE, &restore(&bad, "Restored")),
                restore_pairs(&wildcard(), &table("Restored")),
                "restore {bad:?}"
            );
        }
    }

    #[test]
    fn missing_or_non_string_backup_arn_evaluates_table_wildcard() {
        for input in [
            json!({}),
            json!({"BackupArn": null}),
            json!({"BackupArn": 1}),
        ] {
            for op in ["DescribeBackup", "DeleteBackup"] {
                assert_eq!(
                    targets(op, &input),
                    under(op, &[wildcard()]),
                    "{op} {input}"
                );
            }
        }
        assert_eq!(
            targets(RESTORE, &json!({"TargetTableName": "Restored"})),
            restore_pairs(&wildcard(), &table("Restored"))
        );
        assert_eq!(
            targets(RESTORE, &json!({})),
            restore_pairs(&wildcard(), &wildcard())
        );
    }

    /// `table/*/backup/*`, or `table/<TableName>/backup/*` with a filter.
    /// A `TableName` that is not a string is evaluated as no filter; the
    /// engine then refuses the request.
    #[test]
    fn list_backups_evaluates_the_backup_wildcard_of_the_filter() {
        let op = "ListBackups";
        assert_eq!(targets(op, &json!({})), under(op, &[backups_of(None)]));
        assert_eq!(
            targets(op, &json!({"TableName": "Music"})),
            under(op, &[backups_of(Some("Music"))])
        );
        assert_eq!(
            targets(
                op,
                &json!({"TableName": "Music", "Limit": 5, "BackupType": "ALL"})
            ),
            under(op, &[backups_of(Some("Music"))])
        );
        for input in [
            json!({"TableName": null}),
            json!({"TableName": 7}),
            json!({"TableName": ["Music"]}),
        ] {
            assert_eq!(
                targets(op, &input),
                under(op, &[backups_of(None)]),
                "{input}"
            );
        }
        // The filter is a literal component of the resource, as supplied.
        assert_eq!(
            targets(op, &json!({"TableName": ""})),
            under(
                op,
                &[format!("arn:aws:dynamodb:{REGION}:{ACCT}:table//backup/*")]
            )
        );
        assert_eq!(
            targets(op, &json!({"TableName": "*"})),
            under(op, &[backups_of(None)])
        );
        // Never the bare table wildcard the operation used to be evaluated on.
        for input in [json!({}), json!({"TableName": "Music"})] {
            assert!(!targets(op, &input).contains(&pair(op, &wildcard())));
        }
    }

    #[test]
    fn point_in_time_restore_evaluates_the_source_table() {
        let op = "RestoreTableToPointInTime";
        assert_eq!(
            targets(
                op,
                &json!({"SourceTableName": "Music", "TargetTableName": "R"})
            ),
            under(op, &[table("Music")])
        );
        assert_eq!(
            targets(
                op,
                &json!({"SourceTableArn": table("Music"), "TargetTableName": "R"})
            ),
            under(op, &[wildcard()])
        );
        assert_eq!(
            targets(
                op,
                &json!({"SourceTableName": "Music", "SourceTableArn": table("Music")})
            ),
            under(op, &[table("Music"), wildcard()])
        );
        assert_eq!(
            targets(op, &json!({"TargetTableName": "R"})),
            under(op, &[wildcard()])
        );
    }

    #[test]
    fn table_scoped_backup_operations_stay_on_the_generic_path() {
        for op in [
            "CreateBackup",
            "DescribeContinuousBackups",
            "UpdateContinuousBackups",
            "GetItem",
            "ListTables",
        ] {
            assert!(
                backup_authz_targets(op, &json!({"TableName": "Music"}), REGION, ACCT).is_none(),
                "{op}"
            );
        }
    }

    // ── Policy matcher semantics the model relies on ──────────────────

    #[test]
    fn arn_wildcard_crosses_slash_and_a_table_arn_does_not_cover_its_backups() {
        let b = backup("Music", A);
        assert!(
            arn_match(&wildcard(), &b),
            "table/* must match a backup ARN"
        );
        assert!(arn_match("*", &b));
        assert!(arn_match("arn:aws:dynamodb:*:*:table/*", &b));
        assert!(arn_match(
            &format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/backup/*"),
            &b
        ));
        assert!(
            !arn_match(&table("Music"), &b),
            "table/Music is not table/Music/backup/x"
        );
        assert!(!arn_match(&backup("Music", B), &b));
        assert!(!arn_match(
            &format!("arn:aws:dynamodb:{REGION}:{ACCT}:table/Other/backup/*"),
            &b
        ));
        // The wildcard resource itself is matched by table/* only literally.
        assert!(arn_match(&wildcard(), &wildcard()));
        assert!(!arn_match(&backup("Music", "*"), &wildcard()));
        assert!(!arn_match(&table("Music"), &wildcard()));
    }

    /// The `ListBackups` resources against the statements an operator writes.
    #[test]
    fn backup_wildcard_resources_match_the_statements_that_should_cover_them() {
        let all = backups_of(None);
        let music = backups_of(Some("Music"));
        for value in [&all, &music] {
            assert!(arn_match("*", value));
            assert!(arn_match(&wildcard(), value), "table/* covers {value}");
            assert!(arn_match("arn:aws:dynamodb:*:*:table/*", value));
            assert!(arn_match(&all, value), "table/*/backup/* covers {value}");
            assert!(!arn_match(&table("Music"), value), "table/Music vs {value}");
            assert!(!arn_match(&backups_of(Some("Other")), value));
            assert!(!arn_match(&backup("Music", A), value));
        }
        assert!(arn_match(&music, &music));
        assert!(arn_match(&format!("{}/*", table("Music")), &music));
        assert!(
            !arn_match(&music, &all),
            "a statement on one table's backups does not cover the unfiltered list"
        );
    }

    // ── Decisions ─────────────────────────────────────────────────────

    const BACKUP_OPS: &[&str] = &[
        "dynamodb:CreateBackup",
        "dynamodb:DescribeBackup",
        "dynamodb:DeleteBackup",
        "dynamodb:ListBackups",
        "dynamodb:RestoreTableFromBackup",
        "dynamodb:DescribeContinuousBackups",
        "dynamodb:UpdateContinuousBackups",
        "dynamodb:RestoreTableToPointInTime",
    ];

    /// The item actions a restore needs on its target, as IAM actions.
    const RESTORE_TARGET_OPS: &[&str] = &[
        "dynamodb:Scan",
        "dynamodb:Query",
        "dynamodb:UpdateItem",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:DeleteItem",
        "dynamodb:BatchWriteItem",
    ];

    /// `BACKUP_OPS` plus the restore's dependent actions: what a policy that
    /// means to allow every backup operation has to grant.
    fn all_backup_ops() -> Vec<&'static str> {
        let mut v = BACKUP_OPS.to_vec();
        v.extend_from_slice(RESTORE_TARGET_OPS);
        v
    }

    #[test]
    fn restore_target_ops_mirror_the_evaluated_actions() {
        let from_code: Vec<String> = RESTORE_TARGET_ACTIONS
            .iter()
            .map(|a| format!("dynamodb:{a}"))
            .collect();
        assert_eq!(from_code, RESTORE_TARGET_OPS);
        assert!(!RESTORE_TARGET_ACTIONS.contains(&RESTORE));
    }

    fn every_op_input() -> Vec<(&'static str, Value)> {
        vec![
            (
                "CreateBackup",
                json!({"TableName": "Music", "BackupName": "b"}),
            ),
            ("DescribeBackup", json!({"BackupArn": backup("Music", A)})),
            ("DeleteBackup", json!({"BackupArn": backup("Music", A)})),
            ("ListBackups", json!({})),
            ("ListBackups", json!({"TableName": "Music"})),
            (RESTORE, restore(&backup("Music", A), "Restored")),
            ("DescribeContinuousBackups", json!({"TableName": "Music"})),
            (
                "UpdateContinuousBackups",
                json!({"TableName": "Music", "PointInTimeRecoverySpecification": {}}),
            ),
            (
                "RestoreTableToPointInTime",
                json!({"SourceTableName": "Music", "TargetTableName": "R"}),
            ),
        ]
    }

    #[tokio::test]
    async fn table_wildcard_policy_still_allows_every_backup_operation() {
        let ops = all_backup_ops();
        for resource in [
            wildcard(),
            "arn:aws:dynamodb:*:*:table/*".to_owned(),
            "*".to_owned(),
        ] {
            let (_, cache) = harness(&[("Allow", &ops, &[resource.as_str()])]);
            for (op, input) in every_op_input() {
                let r = decide(&cache, op, &input).await;
                assert!(r.is_ok(), "{op} under Allow {resource}: {r:?}");
            }
        }
    }

    #[tokio::test]
    async fn no_policy_denies_every_backup_operation() {
        let (_, cache) = harness(&[]);
        for (op, input) in every_op_input() {
            assert_denied(&decide(&cache, op, &input).await, op);
        }
    }

    #[tokio::test]
    async fn dev_mode_bypasses_policy_evaluation_for_every_backup_operation() {
        let (store, cache) = harness(&[]);
        let foreign = format!("arn:aws:dynamodb:{REGION}:{OTHER_ACCT}:table/Music/backup/{A}");
        let mut inputs = every_op_input();
        // Another account's ARN passes authz in dev mode; the engine still
        // refuses it with AccessDeniedException.
        inputs.push(("DescribeBackup", json!({"BackupArn": foreign})));
        inputs.push(("DescribeBackup", json!({"BackupArn": "garbage"})));
        for (op, input) in inputs {
            let r = decide_as(&cache, &user(), true, op, &input).await;
            assert!(r.is_ok(), "{op} in dev mode: {r:?}");
        }
        assert!(store.resource_tag_lookups.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn explicit_deny_on_one_backup_denies_only_that_backup() {
        let denied = backup("Music", A);
        let (_, cache) = harness(&[
            ("Allow", &all_backup_ops(), &["*"]),
            (
                "Deny",
                &[
                    "dynamodb:DescribeBackup",
                    "dynamodb:DeleteBackup",
                    "dynamodb:RestoreTableFromBackup",
                ],
                &[denied.as_str()],
            ),
        ]);
        let other = backup("Music", B);
        for op in ["DescribeBackup", "DeleteBackup"] {
            assert_denied(
                &decide(&cache, op, &json!({"BackupArn": denied})).await,
                &format!("{op} of the denied backup"),
            );
            let r = decide(&cache, op, &json!({"BackupArn": other})).await;
            assert!(r.is_ok(), "{op} of another backup: {r:?}");
        }
        assert_denied_on(
            &decide(&cache, RESTORE, &restore(&denied, "Restored")).await,
            RESTORE,
            &denied,
            "restore of the denied backup",
        );
        assert!(
            decide(&cache, RESTORE, &restore(&other, "Restored"))
                .await
                .is_ok()
        );
        // Without a target name the denied backup is still checked first.
        assert_denied(
            &decide(&cache, RESTORE, &json!({"BackupArn": denied})).await,
            "restore of the denied backup without a target name",
        );
    }

    #[tokio::test]
    async fn deny_on_a_table_backups_does_not_cover_another_tables_backups() {
        let (_, cache) = harness(&[
            ("Allow", BACKUP_OPS, &["*"]),
            (
                "Deny",
                &["dynamodb:DescribeBackup"],
                &[&format!(
                    "arn:aws:dynamodb:{REGION}:{ACCT}:table/Music/backup/*"
                )],
            ),
        ]);
        assert_denied(
            &decide(
                &cache,
                "DescribeBackup",
                &json!({"BackupArn": backup("Music", A)}),
            )
            .await,
            "Music backup",
        );
        assert!(
            decide(
                &cache,
                "DescribeBackup",
                &json!({"BackupArn": backup("Other", A)})
            )
            .await
            .is_ok()
        );
    }

    #[tokio::test]
    async fn restore_needs_the_backup_and_the_item_actions_on_the_target() {
        let music_backups = backups_of(Some("Music"));
        let restored = table("Restored");
        let restore_action: &[&str] = &["dynamodb:RestoreTableFromBackup"];

        let (_, both) = harness(&[
            ("Allow", restore_action, &[music_backups.as_str()]),
            ("Allow", RESTORE_TARGET_OPS, &[restored.as_str()]),
        ]);
        let input = restore(&backup("Music", A), "Restored");
        let r = decide(&both, RESTORE, &input).await;
        assert!(r.is_ok(), "backup and target allowed: {r:?}");
        // The same grant in one statement on both resources.
        let (_, joint) = harness(&[(
            "Allow",
            &all_backup_ops(),
            &[music_backups.as_str(), restored.as_str()],
        )]);
        assert!(decide(&joint, RESTORE, &input).await.is_ok());

        let (_, backup_only) = harness(&[("Allow", restore_action, &[music_backups.as_str()])]);
        assert_denied_on(
            &decide(&backup_only, RESTORE, &input).await,
            "Scan",
            &restored,
            "target not allowed: the first dependent action is named",
        );
        let (_, target_only) = harness(&[("Allow", RESTORE_TARGET_OPS, &[restored.as_str()])]);
        assert_denied_on(
            &decide(&target_only, RESTORE, &input).await,
            RESTORE,
            &backup("Music", A),
            "backup not allowed",
        );

        // Other cells under the backup+target allow.
        for (input, what) in [
            (restore(&backup("Music", A), "Elsewhere"), "another target"),
            (
                restore(&backup("Other", A), "Restored"),
                "another table's backup",
            ),
            (
                json!({"BackupArn": backup("Music", A)}),
                "missing target name",
            ),
            (json!({"TargetTableName": "Restored"}), "missing backup ARN"),
            (restore("garbage", "Restored"), "malformed backup ARN"),
        ] {
            assert_denied(&decide(&both, RESTORE, &input).await, what);
        }
    }

    /// Each of the seven dependent actions is required on the target; a
    /// grant of the other six, with or without RestoreTableFromBackup on the
    /// target, is denied naming the missing one.
    #[tokio::test]
    async fn restore_needs_every_dependent_action_on_the_target() {
        let music_backups = backups_of(Some("Music"));
        let restored = table("Restored");
        let input = restore(&backup("Music", A), "Restored");
        for missing in RESTORE_TARGET_OPS {
            let mut granted: Vec<&str> = RESTORE_TARGET_OPS
                .iter()
                .copied()
                .filter(|a| a != missing)
                .collect();
            // RestoreTableFromBackup on the target does not stand in for it.
            granted.push("dynamodb:RestoreTableFromBackup");
            let (_, cache) = harness(&[
                (
                    "Allow",
                    &["dynamodb:RestoreTableFromBackup"],
                    &[music_backups.as_str()],
                ),
                ("Allow", &granted, &[restored.as_str()]),
            ]);
            let action = missing.strip_prefix("dynamodb:").unwrap();
            assert_denied_on(
                &decide(&cache, RESTORE, &input).await,
                action,
                &restored,
                &format!("without {missing} on the target"),
            );
            // Liveness: adding it back allows.
            granted.push(missing);
            let (_, cache) = harness(&[
                (
                    "Allow",
                    &["dynamodb:RestoreTableFromBackup"],
                    &[music_backups.as_str()],
                ),
                ("Allow", &granted, &[restored.as_str()]),
            ]);
            let r = decide(&cache, RESTORE, &input).await;
            assert!(r.is_ok(), "with {missing} restored to the grant: {r:?}");
        }
    }

    /// The breaking change against shipped builds, which evaluated
    /// `RestoreTableFromBackup` on `table/*` and nothing else: a policy
    /// granting only `dynamodb:RestoreTableFromBackup` on `*` (what used to
    /// suffice) no longer restores; the denial names the first dependent
    /// action on the target. A Deny of RestoreTableFromBackup on the target
    /// does not block a restore, a Deny of an item action on it does.
    #[tokio::test]
    async fn restore_action_alone_no_longer_allows_a_restore_and_is_not_evaluated_on_the_target() {
        let input = restore(&backup("Music", A), "Restored");
        let restored = table("Restored");
        let (_, restore_only) = harness(&[("Allow", &["dynamodb:RestoreTableFromBackup"], &["*"])]);
        assert_denied_on(
            &decide(&restore_only, RESTORE, &input).await,
            "Scan",
            &restored,
            "Allow RestoreTableFromBackup on * only",
        );
        // dynamodb:* on * still restores (it covers the dependent actions).
        let (_, star) = harness(&[("Allow", &["dynamodb:*"], &["*"])]);
        assert!(decide(&star, RESTORE, &input).await.is_ok());

        let (_, deny_restore_on_target) = harness(&[
            ("Allow", &all_backup_ops(), &["*"]),
            (
                "Deny",
                &["dynamodb:RestoreTableFromBackup"],
                &[restored.as_str()],
            ),
        ]);
        let r = decide(&deny_restore_on_target, RESTORE, &input).await;
        assert!(
            r.is_ok(),
            "Deny RestoreTableFromBackup on the target is not evaluated: {r:?}"
        );
        for denied_action in RESTORE_TARGET_OPS {
            let (_, cache) = harness(&[
                ("Allow", &all_backup_ops(), &["*"]),
                ("Deny", &[denied_action], &[restored.as_str()]),
            ]);
            assert_denied_on(
                &decide(&cache, RESTORE, &input).await,
                denied_action.strip_prefix("dynamodb:").unwrap(),
                &restored,
                &format!("Deny {denied_action} on the target"),
            );
            // The same Deny on another table does not touch this restore.
            let (_, elsewhere) = harness(&[
                ("Allow", &all_backup_ops(), &["*"]),
                ("Deny", &[denied_action], &[table("Elsewhere").as_str()]),
            ]);
            assert!(decide(&elsewhere, RESTORE, &input).await.is_ok());
        }
    }

    #[tokio::test]
    async fn restore_as_role_session_uses_the_same_pairs() {
        let music_backups = backups_of(Some("Music"));
        let restored = table("Restored");
        let (_, both) = harness(&[
            (
                "Allow",
                &["dynamodb:RestoreTableFromBackup"],
                &[music_backups.as_str()],
            ),
            ("Allow", RESTORE_TARGET_OPS, &[restored.as_str()]),
        ]);
        let input = restore(&backup("Music", A), "Restored");
        assert!(
            decide_as(&both, &role(), false, RESTORE, &input)
                .await
                .is_ok()
        );
        let (_, backup_only) = harness(&[(
            "Allow",
            &["dynamodb:RestoreTableFromBackup"],
            &[music_backups.as_str()],
        )]);
        assert_denied_on(
            &decide_as(&backup_only, &role(), false, RESTORE, &input).await,
            "Scan",
            &restored,
            "role: target not allowed",
        );
        let (_, restore_only) = harness(&[("Allow", &["dynamodb:RestoreTableFromBackup"], &["*"])]);
        assert_denied_on(
            &decide_as(&restore_only, &role(), false, RESTORE, &input).await,
            "Scan",
            &restored,
            "role: restore action alone",
        );
    }

    #[tokio::test]
    async fn table_arn_alone_does_not_allow_operations_on_its_backups() {
        // Behaviour change: before, these were evaluated against table/* and
        // denied by this policy too; now they are evaluated against the backup
        // ARN, which table/Music does not match either. A policy that should
        // cover a table's backups names table/Music/backup/*.
        let music = table("Music");
        let (_, cache) = harness(&[("Allow", &all_backup_ops(), &[music.as_str()])]);
        for op in ["DescribeBackup", "DeleteBackup"] {
            assert_denied(
                &decide(&cache, op, &json!({"BackupArn": backup("Music", A)})).await,
                op,
            );
        }
        // Nor ListBackups filtered to it (table/Music/backup/*), nor a restore
        // of its backup into itself.
        assert_denied(
            &decide(&cache, "ListBackups", &json!({"TableName": "Music"})).await,
            "ListBackups TableName=Music",
        );
        assert_denied_on(
            &decide(&cache, RESTORE, &restore(&backup("Music", A), "Music")).await,
            RESTORE,
            &backup("Music", A),
            "restore of Music's backup into Music",
        );
        // Table-scoped operations on Music are allowed by it.
        for (op, input) in [
            (
                "CreateBackup",
                json!({"TableName": "Music", "BackupName": "b"}),
            ),
            ("DescribeContinuousBackups", json!({"TableName": "Music"})),
            ("UpdateContinuousBackups", json!({"TableName": "Music"})),
            (
                "RestoreTableToPointInTime",
                json!({"SourceTableName": "Music", "TargetTableName": "R"}),
            ),
        ] {
            let r = decide(&cache, op, &input).await;
            assert!(r.is_ok(), "{op}: {r:?}");
        }
        // ...and not for another table.
        assert_denied(
            &decide(&cache, "CreateBackup", &json!({"TableName": "Other"})).await,
            "CreateBackup of Other",
        );
    }

    /// `ListBackups` is granted and denied per table through the backup
    /// wildcard of the filter. A statement on `table/Music/backup/*` allows
    /// the list filtered to Music and nothing else; `table/Music` allows
    /// nothing (shipped builds evaluated the filtered list on `table/Music`,
    /// so such a grant used to list Music's backups); `table/*` and
    /// `table/*/backup/*` allow every list.
    #[tokio::test]
    async fn list_backups_is_granted_per_table_through_the_backup_wildcard() {
        let op = "ListBackups";
        let action: &[&str] = &["dynamodb:ListBackups"];
        let music_filter = json!({"TableName": "Music"});
        let other_filter = json!({"TableName": "Other"});
        let unfiltered = json!({});

        let music_backups = backups_of(Some("Music"));
        let (_, scoped) = harness(&[("Allow", action, &[music_backups.as_str()])]);
        let r = decide(&scoped, op, &music_filter).await;
        assert!(
            r.is_ok(),
            "filtered to Music under Allow {music_backups}: {r:?}"
        );
        assert_denied_on(
            &decide(&scoped, op, &other_filter).await,
            op,
            &backups_of(Some("Other")),
            "filtered to Other",
        );
        assert_denied_on(
            &decide(&scoped, op, &unfiltered).await,
            op,
            &backups_of(None),
            "unfiltered under a per-table grant",
        );
        // A non-string filter is the unfiltered resource.
        assert_denied(
            &decide(&scoped, op, &json!({"TableName": 7})).await,
            "non-string filter",
        );

        // The table's own ARN does not cover its backups (the service's shape
        // is the same: ListBackups has no `table` resource).
        let music = table("Music");
        let (_, table_scoped) = harness(&[("Allow", action, &[music.as_str()])]);
        assert_denied(
            &decide(&table_scoped, op, &music_filter).await,
            "ListBackups under Allow table/Music",
        );
        assert_denied(
            &decide(&table_scoped, op, &unfiltered).await,
            "unfiltered under Allow table/Music",
        );
        // `table/Music/*` does.
        let music_tree = format!("{music}/*");
        let (_, tree) = harness(&[("Allow", action, &[music_tree.as_str()])]);
        assert!(decide(&tree, op, &music_filter).await.is_ok());
        assert_denied(
            &decide(&tree, op, &unfiltered).await,
            "unfiltered under table/Music/*",
        );

        for all in [
            "*".to_owned(),
            wildcard(),
            "arn:aws:dynamodb:*:*:table/*".to_owned(),
            backups_of(None),
            "arn:aws:dynamodb:*:*:table/*/backup/*".to_owned(),
        ] {
            let (_, cache) = harness(&[("Allow", action, &[all.as_str()])]);
            for input in [&music_filter, &other_filter, &unfiltered] {
                let r = decide(&cache, op, input).await;
                assert!(r.is_ok(), "{input} under Allow {all}: {r:?}");
            }
        }

        // A Deny on one table's backups denies the list filtered to it and
        // nothing else (the unfiltered list is `table/*/backup/*`, which the
        // Deny's resource does not match).
        let (_, denied) = harness(&[
            ("Allow", action, &["*"]),
            ("Deny", action, &[music_backups.as_str()]),
        ]);
        assert_denied(
            &decide(&denied, op, &music_filter).await,
            "Deny Music backups",
        );
        assert!(decide(&denied, op, &other_filter).await.is_ok());
        assert!(decide(&denied, op, &unfiltered).await.is_ok());
        // A Deny on table/* covers every list.
        let (_, denied_all) = harness(&[
            ("Allow", action, &["*"]),
            ("Deny", action, &[wildcard().as_str()]),
        ]);
        for input in [&music_filter, &other_filter, &unfiltered] {
            assert_denied(&decide(&denied_all, op, input).await, "Deny table/*");
        }
    }

    #[tokio::test]
    async fn another_accounts_backup_is_denied_and_its_tags_are_not_read() {
        let foreign = format!("arn:aws:dynamodb:{REGION}:{OTHER_ACCT}:table/Music/backup/{A}");
        let (store, cache) = harness(&[("Allow", &all_backup_ops(), &["*"])]);
        for (op, input) in [
            ("DescribeBackup", json!({"BackupArn": foreign})),
            ("DeleteBackup", json!({"BackupArn": foreign})),
            (RESTORE, restore(&foreign, "Restored")),
        ] {
            assert_denied(&decide(&cache, op, &input).await, op);
        }
        assert!(
            store.resource_tag_lookups.lock().unwrap().is_empty(),
            "no resource lookups for a refused request: {:?}",
            store.resource_tag_lookups.lock().unwrap()
        );
        // Liveness: an own-account backup under the same policy is evaluated
        // and allowed. (Backups have no tags, so none are read for it either;
        // see `stored_backup_tags_do_not_satisfy_a_tag_condition`.)
        let own = backup("Music", A);
        assert!(
            decide(&cache, "DescribeBackup", &json!({"BackupArn": own}))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn malformed_or_missing_backup_arn_is_no_more_permissive_than_table_wildcard() {
        // Under a backup-scoped allow, a request naming no valid backup is
        // denied (it would need table/*). Under table/* it reaches the engine,
        // which answers ValidationException.
        let music_backups = backups_of(Some("Music"));
        let action: &[&str] = &["dynamodb:DescribeBackup", "dynamodb:DeleteBackup"];
        let (_, scoped) = harness(&[("Allow", action, &[music_backups.as_str()])]);
        let (_, all) = harness(&[("Allow", action, &[wildcard().as_str()])]);
        let mut inputs: Vec<Value> = malformed_arns()
            .into_iter()
            .map(|a| json!({"BackupArn": a}))
            .collect();
        inputs.push(json!({}));
        inputs.push(json!({"BackupArn": 5}));
        for input in &inputs {
            for op in ["DescribeBackup", "DeleteBackup"] {
                assert_denied(&decide(&scoped, op, input).await, &format!("{op} {input}"));
                let r = decide(&all, op, input).await;
                assert!(r.is_ok(), "{op} {input} under table/*: {r:?}");
            }
        }
        // Liveness for the scoped policy: a well-formed Music backup passes.
        assert!(
            decide(
                &scoped,
                "DescribeBackup",
                &json!({"BackupArn": backup("Music", A)})
            )
            .await
            .is_ok()
        );
    }

    #[tokio::test]
    async fn malformed_backup_arn_still_honours_a_deny_on_table_wildcard() {
        let (_, cache) = harness(&[
            ("Allow", BACKUP_OPS, &["*"]),
            ("Deny", &["dynamodb:DescribeBackup"], &[wildcard().as_str()]),
        ]);
        assert_denied(
            &decide(&cache, "DescribeBackup", &json!({"BackupArn": "garbage"})).await,
            "malformed under Deny table/*",
        );
        // table/* also covers every backup ARN.
        assert_denied(
            &decide(
                &cache,
                "DescribeBackup",
                &json!({"BackupArn": backup("Music", A)}),
            )
            .await,
            "backup under Deny table/*",
        );
    }

    // ── A backup ARN in another region: both spellings ────────────────

    /// The inputs that name Music's backup `A` by its other-region spelling,
    /// per operation.
    fn other_region_inputs() -> Vec<(&'static str, Value)> {
        let elsewhere = backup_elsewhere("Music", A);
        vec![
            ("DescribeBackup", json!({"BackupArn": elsewhere})),
            ("DeleteBackup", json!({"BackupArn": elsewhere})),
            (RESTORE, restore(&elsewhere, "Restored")),
        ]
    }

    /// A Deny on the ARN rebuilt in the server's region denies the
    /// other-region spelling (the engine would resolve the rebuilt ARN first
    /// and act on that backup), and a Deny on the other-region spelling
    /// denies it too (the engine would fall back to it). Neither Deny
    /// touches a backup of another table or id.
    #[tokio::test]
    async fn a_deny_on_either_spelling_denies_the_other_region_backup_arn() {
        let canonical = backup("Music", A);
        let elsewhere = backup_elsewhere("Music", A);
        for (denied, named) in [(&canonical, &canonical), (&elsewhere, &elsewhere)] {
            let (_, cache) = harness(&[
                ("Allow", &all_backup_ops(), &["*"]),
                ("Deny", BACKUP_ARN_OPS, &[denied.as_str()]),
            ]);
            for (op, input) in other_region_inputs() {
                assert_denied_on(
                    &decide(&cache, op, &input).await,
                    op,
                    named,
                    &format!("{op} of the other-region spelling under Deny {denied}"),
                );
            }
            // Another backup of the same table in the other region is not
            // covered by either Deny.
            let r = decide(
                &cache,
                "DescribeBackup",
                &json!({"BackupArn": backup_elsewhere("Music", B)}),
            )
            .await;
            assert!(r.is_ok(), "other id under Deny {denied}: {r:?}");
        }
        // A Deny on the other-region spelling alone does not deny the
        // server-region ARN: that request names one spelling only.
        let (_, cache) = harness(&[
            ("Allow", &all_backup_ops(), &["*"]),
            ("Deny", BACKUP_ARN_OPS, &[elsewhere.as_str()]),
        ]);
        let r = decide(&cache, "DescribeBackup", &json!({"BackupArn": canonical})).await;
        assert!(r.is_ok(), "{r:?}");
        // With both spellings denied, the denial names the rebuilt ARN: it
        // is evaluated first, as the engine resolves it first.
        let (_, cache) = harness(&[
            ("Allow", &all_backup_ops(), &["*"]),
            (
                "Deny",
                BACKUP_ARN_OPS,
                &[elsewhere.as_str(), canonical.as_str()],
            ),
        ]);
        for (op, input) in other_region_inputs() {
            assert_denied_on(
                &decide(&cache, op, &input).await,
                op,
                &canonical,
                &format!("{op} with both spellings denied"),
            );
        }
    }

    /// Both spellings must be allowed: an Allow that names only the
    /// server-region ARN, or only the other-region one, is denied naming the
    /// spelling it leaves out, in evaluation order (the rebuilt ARN first).
    /// A statement with a `*` region, or on `table/*`, covers both.
    #[tokio::test]
    async fn an_allow_must_cover_both_spellings_of_an_other_region_backup_arn() {
        let canonical = backup("Music", A);
        let elsewhere = backup_elsewhere("Music", A);
        let restored = table("Restored");
        let target_grant: Stmt<'_> = ("Allow", RESTORE_TARGET_OPS, &[restored.as_str()]);

        let (_, canonical_only) = harness(&[
            ("Allow", BACKUP_ARN_OPS, &[canonical.as_str()]),
            target_grant,
        ]);
        let (_, elsewhere_only) = harness(&[
            ("Allow", BACKUP_ARN_OPS, &[elsewhere.as_str()]),
            target_grant,
        ]);
        for (op, input) in other_region_inputs() {
            assert_denied_on(
                &decide(&canonical_only, op, &input).await,
                op,
                &elsewhere,
                &format!("{op}: Allow on the server-region ARN only"),
            );
            assert_denied_on(
                &decide(&elsewhere_only, op, &input).await,
                op,
                &canonical,
                &format!("{op}: Allow on the other-region ARN only"),
            );
            // The same requests on the server-region spelling pass under the
            // canonical-only grant: one spelling, allowed.
            let own = input
                .as_object()
                .map(|o| {
                    let mut o = o.clone();
                    o.insert("BackupArn".into(), json!(canonical));
                    Value::Object(o)
                })
                .unwrap();
            let r = decide(&canonical_only, op, &own).await;
            assert!(r.is_ok(), "{op} on the server-region ARN: {r:?}");
        }
        let any_region = format!("arn:aws:dynamodb:*:{ACCT}:table/Music/backup/{A}");
        let any_backup_of_music = format!("arn:aws:dynamodb:*:{ACCT}:table/Music/backup/*");
        for covering in [
            vec![canonical.as_str(), elsewhere.as_str()],
            vec![any_region.as_str()],
            vec![any_backup_of_music.as_str()],
            vec!["arn:aws:dynamodb:*:*:table/*"],
            vec!["*"],
        ] {
            let (_, cache) =
                harness(&[("Allow", BACKUP_ARN_OPS, covering.as_slice()), target_grant]);
            for (op, input) in other_region_inputs() {
                let r = decide(&cache, op, &input).await;
                assert!(r.is_ok(), "{op} under Allow {covering:?}: {r:?}");
            }
        }
        // `table/*` spelled with the server's region covers the rebuilt ARN
        // only, so the other-region spelling is denied under it (shipped
        // builds evaluated these operations on that resource alone and
        // allowed the request); `arn:aws:dynamodb:*:*:table/*` covers both.
        let (_, server_region_tables) = harness(&[
            ("Allow", BACKUP_ARN_OPS, &[wildcard().as_str()]),
            target_grant,
        ]);
        for (op, input) in other_region_inputs() {
            assert_denied_on(
                &decide(&server_region_tables, op, &input).await,
                op,
                &elsewhere,
                &format!("{op} under Allow {}", wildcard()),
            );
        }
    }

    // ── Condition keys in the target evaluations ──────────────────────

    /// The request-level condition keys (`aws:SourceIp`, `aws:SecureTransport`,
    /// `aws:RequestedRegion`, `aws:UserAgent`) are evaluated on the backup
    /// operations as on any other request.
    #[tokio::test]
    async fn backup_operations_evaluate_the_request_level_condition_keys() {
        // An Allow of everything from one network. Before the request context
        // reached the backup branch, aws:SourceIp was absent there, so this
        // allowed GetItem from the network and denied every backup operation.
        let b = backup("Music", A);
        let from_network = |key: &str, condition: Value| {
            vec![json!({
                "Effect": "Allow", "Action": "dynamodb:*", "Resource": "*",
                "Condition": { key: condition },
            })]
        };
        let ops: [(&str, Value); 6] = [
            ("DescribeBackup", json!({"BackupArn": b})),
            ("DeleteBackup", json!({"BackupArn": b})),
            ("ListBackups", json!({})),
            (
                "CreateBackup",
                json!({"TableName": table("Music"), "BackupName": "x"}),
            ),
            (RESTORE, restore(&b, "Restored")),
            ("GetItem", json!({"TableName": "Music", "Key": {}})),
        ];
        for (key, condition) in [
            ("IpAddress", json!({"aws:SourceIp": "10.0.0.0/8"})),
            ("Bool", json!({"aws:SecureTransport": "true"})),
            ("StringEquals", json!({"aws:RequestedRegion": REGION})),
            ("StringLike", json!({"aws:UserAgent": "aws-sdk-*"})),
        ] {
            let (_, cache) = harness_with_tags(from_network(key, condition), Vec::new());
            for (op, input) in &ops {
                let inside = decide_from(&cache, &user(), false, Some("10.1.1.1"), op, input).await;
                assert!(inside.is_ok(), "{key} {op} from 10.1.1.1: {inside:?}");
            }
        }
        let (_, cache) = harness_with_tags(
            from_network("IpAddress", json!({"aws:SourceIp": "10.0.0.0/8"})),
            Vec::new(),
        );
        for (op, input) in &ops {
            assert_denied(
                &decide_from(&cache, &user(), false, Some("192.0.2.1"), op, input).await,
                &format!("{op} from outside the network"),
            );
        }
    }

    /// The seven item-action evaluations on the restore target carry no
    /// `DynamoDB` condition key (`dynamodb:LeadingKeys`, `dynamodb:Attributes`,
    /// `dynamodb:Select`, `dynamodb:ReturnValues`,
    /// `dynamodb:ReturnConsumedCapacity`, `dynamodb:FullTableScan`,
    /// `dynamodb:EnclosingOperation`): a `ForAllValues:*` or `Null: true`
    /// condition on them is satisfied vacuously, so an Allow so conditioned
    /// grants the restore and a Deny so conditioned blocks it, and a bare
    /// `StringEquals` or `Bool` condition is not satisfied, so an Allow so
    /// conditioned does not grant and a Deny so conditioned does not block.
    /// The keys the service sets in these evaluations were not measured.
    #[tokio::test]
    async fn restore_target_evaluations_carry_no_dynamodb_condition_keys() {
        let b = backup("Music", A);
        let input = restore(&b, "Restored");
        let restored = table("Restored");
        let restore_grant = json!({
            "Effect": "Allow", "Action": "dynamodb:RestoreTableFromBackup",
            "Resource": backups_of(Some("Music")),
        });
        let target_grant = |condition: Value| {
            json!({
                "Effect": "Allow", "Action": RESTORE_TARGET_OPS, "Resource": restored,
                "Condition": condition,
            })
        };
        let decide_with = {
            let input = &input;
            move |statements: Vec<Value>| async move {
                let (_, cache) = harness_with_tags(statements, Vec::new());
                decide(&cache, RESTORE, input).await
            }
        };

        // Vacuously satisfied: the row-level allowlisting pattern grants the
        // whole restore.
        for (what, condition) in [
            (
                "ForAllValues:StringEquals LeadingKeys",
                json!({"ForAllValues:StringEquals": {"dynamodb:LeadingKeys": ["alice"]}}),
            ),
            (
                "ForAllValues:StringEquals Attributes",
                json!({"ForAllValues:StringEquals": {"dynamodb:Attributes": ["pk", "name"]}}),
            ),
            (
                "ForAllValues:StringEquals LeadingKeys with a principal variable",
                json!({"ForAllValues:StringEquals": {"dynamodb:LeadingKeys": ["${aws:username}"]}}),
            ),
            (
                "Null LeadingKeys true",
                json!({"Null": {"dynamodb:LeadingKeys": "true"}}),
            ),
            (
                "Null FullTableScan true",
                json!({"Null": {"dynamodb:FullTableScan": "true"}}),
            ),
            (
                "StringEqualsIfExists Select",
                json!({"StringEqualsIfExists": {"dynamodb:Select": "SPECIFIC_ATTRIBUTES"}}),
            ),
        ] {
            let r = decide_with(vec![restore_grant.clone(), target_grant(condition)]).await;
            assert!(r.is_ok(), "{what}: expected the restore allowed, got {r:?}");
        }
        // Not satisfied: the Allow does not grant; the first target pair is
        // named.
        for (what, condition) in [
            (
                "StringEquals LeadingKeys",
                json!({"StringEquals": {"dynamodb:LeadingKeys": "alice"}}),
            ),
            (
                "ForAnyValue:StringEquals LeadingKeys",
                json!({"ForAnyValue:StringEquals": {"dynamodb:LeadingKeys": ["alice"]}}),
            ),
            (
                "Bool FullTableScan false",
                json!({"Bool": {"dynamodb:FullTableScan": "false"}}),
            ),
            (
                "StringEquals Select",
                json!({"StringEquals": {"dynamodb:Select": "SPECIFIC_ATTRIBUTES"}}),
            ),
            (
                "StringEquals ReturnValues",
                json!({"StringEquals": {"dynamodb:ReturnValues": "NONE"}}),
            ),
            (
                "StringEquals ReturnConsumedCapacity",
                json!({"StringEquals": {"dynamodb:ReturnConsumedCapacity": "NONE"}}),
            ),
            (
                "StringEquals EnclosingOperation",
                json!({"StringEquals": {"dynamodb:EnclosingOperation": "RestoreTableFromBackup"}}),
            ),
            (
                "Null LeadingKeys false",
                json!({"Null": {"dynamodb:LeadingKeys": "false"}}),
            ),
        ] {
            assert_denied_on(
                &decide_with(vec![restore_grant.clone(), target_grant(condition)]).await,
                "Scan",
                &restored,
                what,
            );
        }
        // The same keys on a Deny: a `Bool FullTableScan: true` Deny on Scan
        // (the common no-full-scan policy) does not block a restore; a
        // `ForAllValues` or `Null: true` Deny does.
        let allow_all = json!({"Effect": "Allow", "Action": "dynamodb:*", "Resource": "*"});
        let deny_scan = |condition: Value| {
            json!({
                "Effect": "Deny", "Action": "dynamodb:Scan", "Resource": "*",
                "Condition": condition,
            })
        };
        let r = decide_with(vec![
            allow_all.clone(),
            deny_scan(json!({"Bool": {"dynamodb:FullTableScan": "true"}})),
        ])
        .await;
        assert!(r.is_ok(), "Deny Scan if FullTableScan: {r:?}");
        for (what, condition) in [
            (
                "Deny ForAllValues:StringEquals LeadingKeys",
                json!({"ForAllValues:StringEquals": {"dynamodb:LeadingKeys": ["x"]}}),
            ),
            (
                "Deny Null FullTableScan true",
                json!({"Null": {"dynamodb:FullTableScan": "true"}}),
            ),
        ] {
            assert_denied_on(
                &decide_with(vec![allow_all.clone(), deny_scan(condition)]).await,
                "Scan",
                &restored,
                what,
            );
        }
        // Liveness: a plain Scan request under the no-full-scan Deny is
        // blocked, so the key is set on the generic path and absent here.
        let (_, cache) = harness_with_tags(
            vec![
                allow_all,
                deny_scan(json!({"Bool": {"dynamodb:FullTableScan": "true"}})),
            ],
            Vec::new(),
        );
        assert_denied(
            &decide(&cache, "Scan", &json!({"TableName": "Restored"})).await,
            "plain Scan under Deny Scan if FullTableScan",
        );
    }

    // ── CreateBackup with TableName as the table's ARN ────────────────

    /// `CreateBackup {TableName: <Music's ARN>}` is decided on `table/Music`:
    /// an Allow on `table/Music` allows it (in any region of the caller's
    /// account), a Deny on `table/Music` denies it, and an Allow on another
    /// table does not cover it.
    #[tokio::test]
    async fn create_backup_with_a_table_arn_is_decided_on_the_table_it_names() {
        let op = "CreateBackup";
        let action: &[&str] = &["dynamodb:CreateBackup"];
        let music = table("Music");
        let as_arn = |arn: &str| json!({"TableName": arn, "BackupName": "b"});
        let spellings = [
            music.clone(),
            format!("arn:aws:dynamodb:eu-west-1:{ACCT}:table/Music"),
        ];

        let (_, allowed) = harness(&[("Allow", action, &[music.as_str()])]);
        for arn in &spellings {
            let r = decide(&allowed, op, &as_arn(arn)).await;
            assert!(r.is_ok(), "Allow table/Music, TableName {arn}: {r:?}");
        }
        // The same grant does not cover another table's ARN, nor a backup
        // ARN or another account's table (both `table/*`).
        for other in [
            table("Other"),
            backup("Music", A),
            format!("arn:aws:dynamodb:{REGION}:{OTHER_ACCT}:table/Music"),
        ] {
            assert_denied(
                &decide(&allowed, op, &as_arn(&other)).await,
                &format!("Allow table/Music, TableName {other}"),
            );
        }

        let (_, denied) = harness(&[
            ("Allow", action, &["*"]),
            ("Deny", action, &[music.as_str()]),
        ]);
        for arn in &spellings {
            assert_denied_on(
                &decide(&denied, op, &as_arn(arn)).await,
                op,
                &music,
                &format!("Deny table/Music, TableName {arn}"),
            );
        }
        // The Deny does not reach another table's ARN; the plain-name form
        // is decided on the same resource.
        assert!(decide(&denied, op, &as_arn(&table("Other"))).await.is_ok());
        assert_denied_on(
            &decide(
                &denied,
                op,
                &json!({"TableName": "Music", "BackupName": "b"}),
            )
            .await,
            op,
            &music,
            "Deny table/Music, plain TableName",
        );
    }

    // ── Resource tags on backups ──────────────────────────────────────

    fn tag(k: &str, v: &str) -> Vec<(String, String)> {
        vec![(k.to_owned(), v.to_owned())]
    }

    /// One statement on `*` with `StringEquals dynamodb:ResourceTag/classification = value`.
    fn tag_conditioned(effect: &str, actions: &[&str], value: &str) -> Value {
        json!({
            "Effect": effect, "Action": actions, "Resource": "*",
            "Condition": {"StringEquals": {"dynamodb:ResourceTag/classification": value}},
        })
    }

    fn harness_with_tags(
        statements: Vec<Value>,
        tags: Vec<(String, Vec<(String, String)>)>,
    ) -> (Arc<Store>, CachedAuthzStore) {
        let doc = json!({"Version": "2012-10-17", "Statement": statements}).to_string();
        let store = Arc::new(Store {
            policies: vec![doc],
            resource_tags: tags.into_iter().collect(),
            ..Store::default()
        });
        let cache = CachedAuthzStore::pass_through(store.clone(), AuthzCacheConfig::default());
        (store, cache)
    }

    const BACKUP_ARN_OPS: &[&str] = &[
        "dynamodb:DescribeBackup",
        "dynamodb:DeleteBackup",
        "dynamodb:RestoreTableFromBackup",
    ];

    /// Tags stored under a backup ARN (TagResource accepted them before it
    /// refused backups) do not feed `dynamodb:ResourceTag`: the service has no
    /// tag condition key for backups. Same for a user and a role session.
    #[tokio::test]
    async fn stored_backup_tags_do_not_satisfy_a_tag_condition() {
        let b = backup("Music", A);
        let mut actions = BACKUP_ARN_OPS.to_vec();
        actions.push("dynamodb:DescribeTable");
        actions.extend_from_slice(RESTORE_TARGET_OPS);
        let (store, cache) = harness_with_tags(
            vec![tag_conditioned("Allow", &actions, "dev")],
            vec![
                (b.clone(), tag("classification", "dev")),
                (table("Music"), tag("classification", "dev")),
                (table("Restored"), tag("classification", "dev")),
            ],
        );
        for identity in [user(), role()] {
            // Liveness: the same condition allows a table whose stored tags match.
            let r = decide_as(
                &cache,
                &identity,
                false,
                "DescribeTable",
                &json!({"TableName": "Music"}),
            )
            .await;
            assert!(r.is_ok(), "{identity:?}: tagged table: {r:?}");
            for (op, input) in [
                ("DescribeBackup", json!({"BackupArn": b})),
                ("DeleteBackup", json!({"BackupArn": b})),
                (RESTORE, restore(&b, "Restored")),
            ] {
                assert_denied(
                    &decide_as(&cache, &identity, false, op, &input).await,
                    &format!("{identity:?} {op} of a backup with stored tags"),
                );
            }
        }
        let lookups = store.resource_tag_lookups.lock().unwrap().clone();
        assert!(lookups.contains(&table("Music")), "{lookups:?}");
        assert!(
            lookups
                .iter()
                .all(|a| !extenddb_storage::util::is_backup_resource_arn(a)),
            "backup tags were read: {lookups:?}"
        );
    }

    /// The dependent item actions evaluated on `table/<TargetTableName>`
    /// read that ARN's resource tags like any item operation would. In a
    /// served deployment a target that does not exist has no tags, so a
    /// tag-conditioned Allow finds none; a target that exists is evaluated
    /// with its tags, and the restore then fails `TableAlreadyExistsException`
    /// (or `TableInUseException` for a `CREATING` restore target). This pins
    /// the mechanism with a store that holds tags under the name.
    #[tokio::test]
    async fn target_table_tags_condition_the_dependent_actions_of_a_restore() {
        let b = backup("Music", A);
        let (store, cache) = harness_with_tags(
            vec![
                json!({"Effect": "Allow", "Action": "dynamodb:RestoreTableFromBackup",
                       "Resource": backups_of(Some("Music"))}),
                tag_conditioned("Allow", RESTORE_TARGET_OPS, "dev"),
            ],
            vec![(table("Dev"), tag("classification", "dev"))],
        );
        for identity in [user(), role()] {
            let r = decide_as(&cache, &identity, false, RESTORE, &restore(&b, "Dev")).await;
            assert!(r.is_ok(), "{identity:?}: tagged target: {r:?}");
            assert_denied_on(
                &decide_as(&cache, &identity, false, RESTORE, &restore(&b, "Plain")).await,
                "Scan",
                &table("Plain"),
                &format!("{identity:?}: untagged target"),
            );
        }
        let lookups = store.resource_tag_lookups.lock().unwrap().clone();
        assert!(lookups.contains(&table("Dev")), "{lookups:?}");
        assert!(lookups.contains(&table("Plain")), "{lookups:?}");
    }

    /// The other side of the same rule: a tag-conditioned Deny does not match
    /// a backup either, whatever is stored under its ARN, as on the service
    /// where the key is absent for backups. A Deny that must cover backups
    /// names their ARNs (`table/<name>/backup/*`).
    #[tokio::test]
    async fn a_tag_conditioned_deny_does_not_match_a_backup() {
        let b = backup("Music", A);
        let (_, cache) = harness_with_tags(
            vec![
                json!({"Effect": "Allow", "Action": BACKUP_ARN_OPS, "Resource": "*"}),
                tag_conditioned("Deny", BACKUP_ARN_OPS, "prod"),
            ],
            vec![(b.clone(), tag("classification", "prod"))],
        );
        let r = decide(&cache, "DescribeBackup", &json!({"BackupArn": b})).await;
        assert!(r.is_ok(), "{r:?}");
    }
}
