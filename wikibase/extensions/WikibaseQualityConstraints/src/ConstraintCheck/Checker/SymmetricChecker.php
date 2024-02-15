<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Services\Lookup\EntityLookup;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementListProvider;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class SymmetricChecker implements ConstraintChecker {

	/**
	 * @var EntityLookup
	 */
	private $entityLookup;

	/**
	 * @var ConnectionCheckerHelper
	 */
	private $connectionCheckerHelper;

	public function __construct(
		EntityLookup $lookup,
		ConnectionCheckerHelper $connectionCheckerHelper
	) {
		$this->entityLookup = $lookup;
		$this->connectionCheckerHelper = $connectionCheckerHelper;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getSupportedContextTypes() {
		return [
			Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
			// TODO T175594
			Context::TYPE_QUALIFIER => CheckResult::STATUS_TODO,
			Context::TYPE_REFERENCE => CheckResult::STATUS_TODO,
		];
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getDefaultContextTypes() {
		return [
			Context::TYPE_STATEMENT,
			// TODO T175594
			// Context::TYPE_QUALIFIER,
			// Context::TYPE_REFERENCE,
		];
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * Checks 'Symmetric' constraint.
	 *
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @return CheckResult
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		if ( $context->getSnakRank() === Statement::RANK_DEPRECATED ) {
			return new CheckResult( $context, $constraint, CheckResult::STATUS_DEPRECATED );
		}

		$snak = $context->getSnak();
		$propertyId = $context->getSnak()->getPropertyId();

		if ( !$snak instanceof PropertyValueSnak ) {
			// nothing to check
			return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
		}

		$dataValue = $snak->getDataValue();

		/*
		 * error handling:
		 *   type of $dataValue for properties with 'Symmetric' constraint has to be 'wikibase-entityid'
		 */
		if ( !$dataValue instanceof EntityIdValue ) {
			$message = ( new ViolationMessage( 'wbqc-violation-message-value-needed-of-type' ) )
				->withEntityId( new ItemId( $constraint->getConstraintTypeItemId() ), Role::CONSTRAINT_TYPE_ITEM )
				->withDataValueType( 'wikibase-entityid' );
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION, $message );
		}

		$targetEntityId = $dataValue->getEntityId();
		$targetEntity = $this->entityLookup->getEntity( $targetEntityId );
		if ( !$targetEntity instanceof StatementListProvider ) {
			$message = new ViolationMessage( 'wbqc-violation-message-target-entity-must-exist' );
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION, $message );
		}

		$symmetricStatement = $this->connectionCheckerHelper->findStatementWithPropertyAndEntityIdValue(
			$targetEntity->getStatements(),
			$propertyId,
			$context->getEntity()->getId()
		);
		if ( $symmetricStatement !== null ) {
			$message = null;
			$status = CheckResult::STATUS_COMPLIANCE;
		} else {
			$message = ( new ViolationMessage( 'wbqc-violation-message-symmetric' ) )
				->withEntityId( $targetEntityId, Role::SUBJECT )
				->withEntityId( $propertyId, Role::PREDICATE )
				->withEntityId( $context->getEntity()->getId(), Role::OBJECT );
			$status = CheckResult::STATUS_VIOLATION;
		}

		return ( new CheckResult( $context, $constraint, $status, $message ) )
			->withMetadata( Metadata::ofDependencyMetadata(
				DependencyMetadata::ofEntityId( $targetEntityId ) ) );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		// no parameters
		return [];
	}

}
