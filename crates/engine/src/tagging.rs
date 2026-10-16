// Copyright 2026 ExtendDB contributors
// SPDX-License-Identifier: Apache-2.0

//! `TagResource`, `UntagResource`, and `ListTagsOfResource` operation handlers.

use extenddb_core::error::DynamoDbError;
use extenddb_core::types::{
    ListTagsOfResourceInput, ListTagsOfResourceOutput, TagResourceInput, UntagResourceInput,
};
use serde_json::Value;

use crate::OperationContext;
use crate::sanitize_storage_error;
use crate::serialize_output;

/// Split a `DynamoDB` resource ARN into `(region, account, resource)`.
///
/// Expected format: `arn:aws:dynamodb:{region}:{account}:table/{name}[/...]`
fn split_dynamodb_arn(arn: &str) -> Option<(&str, &str, &str)> {
    let rest = arn.strip_prefix("arn:aws:dynamodb:")?;
    let mut parts = rest.splitn(3, ':');
    let region = parts.next()?;
    let account = parts.next()?;
    let resource = parts.next()?;
    Some((region, account, resource))
}

/// Extract the table name from a `DynamoDB` table ARN.
///
/// Expected format: `arn:aws:dynamodb:{region}:{account}:table/{name}[/...]`
fn extract_table_name_from_arn(arn: &str) -> Option<&str> {
    let (_, _, resource) = split_dynamodb_arn(arn)?;
    let table_name = resource.strip_prefix("table/")?;
    // Strip any sub-resource (e.g. /index/foo, /stream/label)
    Some(table_name.split('/').next().unwrap_or(table_name))
}

/// Validate that the ARN refers to an existing table owned by the caller.
///
/// The error classes and messages below match `DynamoDB`, verified against the
/// service, except where marked:
///
/// | Input | Result |
/// |---|---|
/// | Does not start with `arn:` | `ValidationException` (ARNs must start with `arn:`) |
/// | Valid ARN for another service | `AccessDeniedException` |
/// | `DynamoDB` ARN naming a non-table resource | `ValidationException` (not a `DynamoDB` resource arn) |
/// | Account differs from the caller's | `AccessDeniedException` |
/// | Region differs from this deployment's | `ValidationException` (invalid `TableArn`) |
/// | A backup, `table/<name>/backup[/...]` | `ValidationException` (not a `DynamoDB` resource arn); not measured on the service |
/// | Table does not exist | `ResourceNotFoundException` |
///
/// Account is checked before region, matching the service: a cross-account ARN
/// is denied without revealing whether its region is valid.
async fn validate_resource_arn(arn: &str, ctx: &OperationContext) -> Result<(), DynamoDbError> {
    let table_name = check_resource_arn_shape(arn, &ctx.account_id, &ctx.region)?;

    // Verify the table exists via table_key_info (lightweight check).
    ctx.storage
        .table_key_info(&ctx.account_id, table_name)
        .await
        .map_err(|e| match e {
            extenddb_storage::error::StorageError::TableNotFound(_) => {
                DynamoDbError::ResourceNotFoundException(format!(
                    "Requested resource not found: ResourceArn: {arn} not found"
                ))
            }
            other => sanitize_storage_error(other),
        })?;

    Ok(())
}

/// The checks of [`validate_resource_arn`] that need no storage, in its
/// order; returns the table name to look up.
fn check_resource_arn_shape<'a>(
    arn: &'a str,
    account_id: &str,
    region: &str,
) -> Result<&'a str, DynamoDbError> {
    let not_a_resource_arn = || {
        DynamoDbError::ValidationException(format!(
            "One or more parameter values were invalid: \
             Provided Arn is not a DynamoDB resource arn: {arn}"
        ))
    };

    if !arn.starts_with("arn:") {
        return Err(DynamoDbError::ValidationException(format!(
            "One or more parameter values were invalid: ARNs must start with 'arn:': {arn}"
        )));
    }

    // An ARN for another service is denied rather than reported as malformed:
    // authorization is evaluated against the resource before its shape is
    // inspected further.
    let Some((arn_region, arn_account, resource)) = split_dynamodb_arn(arn) else {
        return Err(DynamoDbError::AccessDeniedException(
            "Access is denied".to_owned(),
        ));
    };

    if !resource.starts_with("table/") {
        return Err(not_a_resource_arn());
    }

    if arn_account != account_id {
        return Err(DynamoDbError::AccessDeniedException(
            "Access is denied".to_owned(),
        ));
    }

    if arn_region != region {
        return Err(DynamoDbError::ValidationException(format!(
            "Invalid TableArn: Invalid ResourceArn provided as input {arn}"
        )));
    }

    // Backups take no tags on the service. Tags stored under a backup ARN
    // would otherwise be read by authorization, which evaluates DescribeBackup,
    // DeleteBackup and RestoreTableFromBackup on the backup ARN, so a caller
    // allowed TagResource could satisfy a `dynamodb:ResourceTag` condition on
    // a backup. Refused with the text this function already uses for an ARN
    // shape it does not tag; the service's own reply was not measured.
    if extenddb_storage::util::is_backup_resource_arn(arn) {
        return Err(not_a_resource_arn());
    }

    extract_table_name_from_arn(arn).ok_or_else(not_a_resource_arn)
}

/// Handle `TagResource` — add or overwrite tags on a resource.
///
/// # Errors
///
/// Returns `ResourceNotFoundException` if the resource does not exist.
/// Returns `ValidationException` if the resource ARN is empty.
/// Returns `InternalServerError` on storage failures.
pub async fn handle_tag_resource(
    body: Value,
    ctx: &OperationContext,
) -> Result<Value, DynamoDbError> {
    let input: TagResourceInput = serde_json::from_value(body).map_err(crate::deserialize_error)?;

    if input.resource_arn.is_empty() {
        return Err(DynamoDbError::ValidationException(
            "ResourceArn must not be empty".to_owned(),
        ));
    }

    validate_resource_arn(&input.resource_arn, ctx).await?;

    ctx.storage
        .tag_resource(&input.resource_arn, &input.tags)
        .await
        .map_err(sanitize_storage_error)?;

    // Drop any cached resource-tag entry so the new tags are visible to
    // ABAC policy evaluation immediately.
    ctx.auth_cache
        .invalidate_resource_tags(&input.resource_arn)
        .await;

    // TagResource returns an empty body on success.
    Ok(Value::Object(serde_json::Map::new()))
}

/// Handle `UntagResource` — remove tags by key from a resource.
///
/// # Errors
///
/// Returns `ResourceNotFoundException` if the resource does not exist.
/// Returns `ValidationException` if the resource ARN is empty.
/// Returns `InternalServerError` on storage failures.
pub async fn handle_untag_resource(
    body: Value,
    ctx: &OperationContext,
) -> Result<Value, DynamoDbError> {
    let input: UntagResourceInput =
        serde_json::from_value(body).map_err(crate::deserialize_error)?;

    if input.resource_arn.is_empty() {
        return Err(DynamoDbError::ValidationException(
            "ResourceArn must not be empty".to_owned(),
        ));
    }

    validate_resource_arn(&input.resource_arn, ctx).await?;

    ctx.storage
        .untag_resource(&input.resource_arn, &input.tag_keys)
        .await
        .map_err(sanitize_storage_error)?;

    ctx.auth_cache
        .invalidate_resource_tags(&input.resource_arn)
        .await;

    // UntagResource returns an empty body on success.
    Ok(Value::Object(serde_json::Map::new()))
}

/// Handle `ListTagsOfResource` — list all tags for a resource.
///
/// # Errors
///
/// Returns `ResourceNotFoundException` if the resource does not exist.
/// Returns `ValidationException` if the resource ARN is empty.
/// Returns `InternalServerError` on storage failures.
pub async fn handle_list_tags_of_resource(
    body: Value,
    ctx: &OperationContext,
) -> Result<Value, DynamoDbError> {
    let input: ListTagsOfResourceInput =
        serde_json::from_value(body).map_err(crate::deserialize_error)?;

    if input.resource_arn.is_empty() {
        return Err(DynamoDbError::ValidationException(
            "ResourceArn must not be empty".to_owned(),
        ));
    }

    validate_resource_arn(&input.resource_arn, ctx).await?;

    let tags = ctx
        .storage
        .list_tags(&input.resource_arn)
        .await
        .map_err(sanitize_storage_error)?;

    let output = ListTagsOfResourceOutput {
        tags,
        next_token: None, // All tags returned in one page.
    };
    serialize_output(&output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGION: &str = "us-east-1";
    const ACCT: &str = "123456789012";
    const OTHER_ACCT: &str = "210987654321";
    const BACKUP_ID: &str = "01791976431438-73d8d5bc";

    fn own(resource: &str) -> String {
        format!("arn:aws:dynamodb:{REGION}:{ACCT}:{resource}")
    }

    fn check(arn: &str) -> Result<&str, DynamoDbError> {
        check_resource_arn_shape(arn, ACCT, REGION)
    }

    fn assert_not_a_resource_arn(arn: &str) {
        match check(arn) {
            Err(DynamoDbError::ValidationException(m)) => assert_eq!(
                m,
                format!(
                    "One or more parameter values were invalid: \
                     Provided Arn is not a DynamoDB resource arn: {arn}"
                ),
                "{arn}"
            ),
            other => panic!("{arn}: expected ValidationException, got {other:?}"),
        }
    }

    #[test]
    fn table_and_table_sub_resources_other_than_backups_name_their_table() {
        for (resource, table) in [
            ("table/Music", "Music"),
            ("table/Music/stream/2026-10-19T00:00:00.000", "Music"),
            ("table/Music/index/byArtist", "Music"),
            // A table whose name is `backup`.
            ("table/backup", "backup"),
            ("table/backup/stream/2026-10-19T00:00:00.000", "backup"),
        ] {
            let arn = own(resource);
            assert_eq!(check(&arn).unwrap(), table, "{arn}");
        }
    }

    #[test]
    fn backup_arns_are_refused_before_the_table_lookup() {
        for resource in [
            format!("table/Music/backup/{BACKUP_ID}"),
            // Legacy id shapes still parse as backup ARNs.
            "table/Music/backup/1489602797149-x".to_owned(),
            // Truncated and extended forms.
            "table/Music/backup".to_owned(),
            "table/Music/backup/".to_owned(),
            format!("table/Music/backup/{BACKUP_ID}/x"),
            format!("table/Music/backup/{BACKUP_ID}:x"),
            // A backup of a table named `backup`.
            format!("table/backup/backup/{BACKUP_ID}"),
            format!("table//backup/{BACKUP_ID}"),
        ] {
            assert_not_a_resource_arn(&own(&resource));
        }
    }

    #[test]
    fn earlier_checks_keep_their_order_for_backup_arns() {
        // Another account: denied, as for a table, without saying more.
        let arn = format!("arn:aws:dynamodb:{REGION}:{OTHER_ACCT}:table/Music/backup/{BACKUP_ID}");
        assert!(
            matches!(check(&arn), Err(DynamoDbError::AccessDeniedException(_))),
            "{arn}"
        );
        // Another region: the region error, still a ValidationException.
        let arn = format!("arn:aws:dynamodb:eu-west-1:{ACCT}:table/Music/backup/{BACKUP_ID}");
        match check(&arn) {
            Err(DynamoDbError::ValidationException(m)) => {
                assert!(m.starts_with("Invalid TableArn"), "{m}");
            }
            other => panic!("{arn}: {other:?}"),
        }
        // Another service.
        let arn = format!("arn:aws:s3:{REGION}:{ACCT}:table/Music/backup/{BACKUP_ID}");
        assert!(
            matches!(check(&arn), Err(DynamoDbError::AccessDeniedException(_))),
            "{arn}"
        );
    }

    #[test]
    fn malformed_arns_are_refused() {
        match check("table/Music/backup/1") {
            Err(DynamoDbError::ValidationException(m)) => {
                assert!(m.contains("ARNs must start with 'arn:'"), "{m}");
            }
            other => panic!("{other:?}"),
        }
        for resource in [
            format!("backup/{BACKUP_ID}"),
            "index/foo".to_owned(),
            String::new(),
        ] {
            assert_not_a_resource_arn(&own(&resource));
        }
        // Too few fields to be a DynamoDB ARN at all.
        assert!(matches!(
            check("arn:aws:dynamodb:us-east-1"),
            Err(DynamoDbError::AccessDeniedException(_))
        ));
    }
}
