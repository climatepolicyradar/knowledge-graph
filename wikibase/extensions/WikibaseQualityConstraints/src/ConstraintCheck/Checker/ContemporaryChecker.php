<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use Config;
use DataValues\DataValue;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\EntityLookup;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementList;
use Wikibase\DataModel\Statement\StatementListProvider;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\RangeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author David Abián
 * @license GPL-2.0-or-later
 */
class ContemporaryChecker implements ConstraintChecker {

	/**
	 * @var RangeCheckerHelper
	 */
	private $rangeCheckerHelper;

	/**
	 * @var Config
	 */
	private $config;

	/**
	 * @var EntityLookup
	 */
	private $entityLookup;

	/**
	 * Name of the configuration variable for the array of IDs of the properties that
	 * state the start time of the entities.
	 */
	public const CONFIG_VARIABLE_START_PROPERTY_IDS = 'WBQualityConstraintsStartTimePropertyIds';

	/**
	 * Name of the configuration variable for the array of IDs of the properties that
	 * state the end time of the entities.
	 */
	public const CONFIG_VARIABLE_END_PROPERTY_IDS = 'WBQualityConstraintsEndTimePropertyIds';

	public function __construct(
		EntityLookup $entityLookup,
		RangeCheckerHelper $rangeCheckerHelper,
		Config $config
	) {
		$this->entityLookup = $entityLookup;
		$this->rangeCheckerHelper = $rangeCheckerHelper;
		$this->config = $config;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getSupportedContextTypes() {
		return [
			Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
			Context::TYPE_QUALIFIER => CheckResult::STATUS_NOT_IN_SCOPE,
			Context::TYPE_REFERENCE => CheckResult::STATUS_NOT_IN_SCOPE,
		];
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getDefaultContextTypes() {
		return [ Context::TYPE_STATEMENT ];
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * Checks 'Contemporary' constraint.
	 *
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @return CheckResult
	 * @throws \ConfigException
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		if ( $context->getSnakRank() === Statement::RANK_DEPRECATED ) {
			return new CheckResult( $context, $constraint, CheckResult::STATUS_DEPRECATED );
		}
		$snak = $context->getSnak();
		if ( !$snak instanceof PropertyValueSnak ) {
			// nothing to check
			return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
		}

		$dataValue = $snak->getDataValue();
		if ( !$dataValue instanceof EntityIdValue ) {
			// wrong data type
			$message = ( new ViolationMessage( 'wbqc-violation-message-value-needed-of-type' ) )
				->withEntityId( new ItemId( $constraint->getConstraintTypeItemId() ), Role::CONSTRAINT_TYPE_ITEM )
				->withDataValueType( 'wikibase-entityid' );
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION, $message );
		}

		$objectId = $dataValue->getEntityId();
		$objectItem = $this->entityLookup->getEntity( $objectId );
		if ( !( $objectItem instanceof StatementListProvider ) ) {
			// object was deleted/doesn't exist
			$message = new ViolationMessage( 'wbqc-violation-message-value-entity-must-exist' );
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION, $message );
		}
		/** @var Statement[] $objectStatements */
		$objectStatements = $objectItem->getStatements()->toArray();

		$subjectId = $context->getEntity()->getId();
		$subjectStatements = $context->getEntity()->getStatements()->toArray();
		/** @var String[] $startPropertyIds */
		$startPropertyIds = $this->config->get( self::CONFIG_VARIABLE_START_PROPERTY_IDS );
		/** @var String[] $endPropertyIds */
		$endPropertyIds = $this->config->get( self::CONFIG_VARIABLE_END_PROPERTY_IDS );
		$subjectStartValue = $this->getExtremeValue(
			$startPropertyIds,
			$subjectStatements,
			'start'
		);
		$objectStartValue = $this->getExtremeValue(
			$startPropertyIds,
			$objectStatements,
			'start'
		);
		$subjectEndValue = $this->getExtremeValue(
			$endPropertyIds,
			$subjectStatements,
			'end'
		);
		$objectEndValue = $this->getExtremeValue(
			$endPropertyIds,
			$objectStatements,
			'end'
		);
		if (
			$this->rangeCheckerHelper->getComparison( $subjectStartValue, $subjectEndValue ) <= 0 &&
			$this->rangeCheckerHelper->getComparison( $objectStartValue, $objectEndValue ) <= 0 && (
				$this->rangeCheckerHelper->getComparison( $subjectEndValue, $objectStartValue ) < 0 ||
				$this->rangeCheckerHelper->getComparison( $objectEndValue, $subjectStartValue ) < 0
			)
		) {
			if (
				$subjectEndValue == null ||
				$this->rangeCheckerHelper->getComparison( $objectEndValue, $subjectEndValue ) < 0
			) {
				$earlierEntityId = $objectId;
				$minEndValue = $objectEndValue;
				$maxStartValue = $subjectStartValue;
			} else {
				$earlierEntityId = $subjectId;
				$minEndValue = $subjectEndValue;
				$maxStartValue = $objectStartValue;
			}
			$message = $this->getViolationMessage(
				// @phan-suppress-next-line PhanTypeMismatchArgumentNullable
				$earlierEntityId,
				$subjectId,
				$context->getSnak()->getPropertyId(),
				$objectId,
				// @phan-suppress-next-line PhanTypeMismatchArgumentNullable
				$minEndValue,
				$maxStartValue
			);
			$status = CheckResult::STATUS_VIOLATION;
		} else {
			$message = null;
			$status = CheckResult::STATUS_COMPLIANCE;
		}
		return new CheckResult( $context, $constraint, $status, $message );
	}

	/**
	 * @param string[] $extremePropertyIds
	 * @param Statement[] $statements
	 * @param string $startOrEnd 'start' or 'end'
	 *
	 * @return DataValue|null
	 */
	private function getExtremeValue( $extremePropertyIds, $statements, $startOrEnd ) {
		if ( $startOrEnd !== 'start' && $startOrEnd !== 'end' ) {
			throw new \InvalidArgumentException( '$startOrEnd must be \'start\' or \'end\'.' );
		}
		$extremeValue = null;
		foreach ( $extremePropertyIds as $extremePropertyId ) {
			$statementList = new StatementList( ...$statements );
			$extremeStatements = $statementList->getByPropertyId( new NumericPropertyId( $extremePropertyId ) );
			/** @var Statement $extremeStatement */
			foreach ( $extremeStatements as $extremeStatement ) {
				if ( $extremeStatement->getRank() !== Statement::RANK_DEPRECATED ) {
					$snak = $extremeStatement->getMainSnak();
					if ( !$snak instanceof PropertyValueSnak ) {
						return null;
					} else {
						$comparison = $this->rangeCheckerHelper->getComparison(
							$snak->getDataValue(),
							$extremeValue
						);
						if (
							$extremeValue === null ||
							( $startOrEnd === 'start' && $comparison < 0 ) ||
							( $startOrEnd === 'end' && $comparison > 0 )
						) {
							$extremeValue = $snak->getDataValue();
						}
					}
				}
			}
		}
		return $extremeValue;
	}

	/**
	 * @param EntityId $earlierEntityId
	 * @param EntityId $subjectId
	 * @param EntityId $propertyId
	 * @param EntityId $objectId
	 * @param DataValue $minEndValue
	 * @param DataValue $maxStartValue
	 *
	 * @return ViolationMessage
	 */
	private function getViolationMessage(
		EntityId $earlierEntityId,
		EntityId $subjectId,
		EntityId $propertyId,
		EntityId $objectId,
		DataValue $minEndValue,
		DataValue $maxStartValue
	) {
		$messageKey = $earlierEntityId === $subjectId ?
			'wbqc-violation-message-contemporary-subject-earlier' :
			'wbqc-violation-message-contemporary-value-earlier';
		return ( new ViolationMessage( $messageKey ) )
			->withEntityId( $subjectId, Role::SUBJECT )
			->withEntityId( $propertyId, Role::PREDICATE )
			->withEntityId( $objectId, Role::OBJECT )
			->withDataValue( $minEndValue, Role::OBJECT )
			->withDataValue( $maxStartValue, Role::OBJECT );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		// no parameters
		return [];
	}

}
