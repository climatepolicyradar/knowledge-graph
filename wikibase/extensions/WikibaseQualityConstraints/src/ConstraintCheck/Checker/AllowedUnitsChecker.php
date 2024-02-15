<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\Lib\Units\UnitConverter;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\UnitsParameter;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class AllowedUnitsChecker implements ConstraintChecker {

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	/**
	 * @var UnitConverter|null
	 */
	private $unitConverter;

	/**
	 * @param ConstraintParameterParser $constraintParameterParser
	 * @param UnitConverter|null $unitConverter
	 */
	public function __construct(
		ConstraintParameterParser $constraintParameterParser,
		UnitConverter $unitConverter = null
	) {
		$this->constraintParameterParser = $constraintParameterParser;
		$this->unitConverter = $unitConverter;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getSupportedContextTypes() {
		return self::ALL_CONTEXT_TYPES_SUPPORTED;
	}

	/**
	 * @codeCoverageIgnore This method is purely declarative.
	 */
	public function getDefaultContextTypes() {
		return Context::ALL_CONTEXT_TYPES;
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * Checks an “allowed units” constraint.
	 *
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @throws ConstraintParameterException
	 * @return CheckResult
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		$constraintParameters = $constraint->getConstraintParameters();
		$unitsParameter = $this->constraintParameterParser
			->parseUnitsParameter(
				$constraintParameters,
				$constraint->getConstraintTypeItemId()
			);

		$snak = $context->getSnak();
		if ( !$snak instanceof PropertyValueSnak ) {
			// nothing to check
			return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
		}

		$dataValue = $snak->getDataValue();
		if ( !$dataValue instanceof UnboundedQuantityValue ) {
			$message = ( new ViolationMessage( 'wbqc-violation-message-value-needed-of-type' ) )
				->withEntityId( new ItemId( $constraint->getConstraintTypeItemId() ), Role::CONSTRAINT_TYPE_ITEM )
				->withDataValueType( 'quantity' );
			return new CheckResult( $context, $constraint, CheckResult::STATUS_VIOLATION, $message );
		}

		if ( $dataValue->getUnit() === '1' ) {
			return $this->checkUnitless( $context, $constraint, $unitsParameter, $snak );
		}

		$status = CheckResult::STATUS_VIOLATION;
		$actualUnit = $this->standardize( $dataValue )->getUnit();
		foreach ( $unitsParameter->getUnitQuantities() as $unitQuantity ) {
			$allowedUnit = $this->standardize( $unitQuantity )->getUnit();
			if ( $actualUnit === $allowedUnit ) {
				$status = CheckResult::STATUS_COMPLIANCE;
				break;
			}
		}

		if ( $status === CheckResult::STATUS_VIOLATION ) {
			if ( $unitsParameter->getUnitItemIds() === [] ) {
				$message = ( new ViolationMessage( 'wbqc-violation-message-units-none' ) )
					->withEntityId( $snak->getPropertyId(), Role::CONSTRAINT_PROPERTY );
			} else {
				$messageKey = $unitsParameter->getUnitlessAllowed() ?
					'wbqc-violation-message-units-or-none' :
					'wbqc-violation-message-units';
				$message = ( new ViolationMessage( $messageKey ) )
					->withEntityId( $snak->getPropertyId(), Role::CONSTRAINT_PROPERTY )
					->withEntityIdList( $unitsParameter->getUnitItemIds(), Role::CONSTRAINT_PARAMETER_VALUE );
			}
		} else {
			$message = null;
		}

		return new CheckResult( $context, $constraint, $status, $message );
	}

	/**
	 * @param Context $context
	 * @param Constraint $constraint
	 * @param UnitsParameter $unitsParameter
	 * @param PropertyValueSnak $snak
	 * @return CheckResult
	 */
	private function checkUnitless(
		Context $context,
		Constraint $constraint,
		UnitsParameter $unitsParameter,
		PropertyValueSnak $snak
	) {
		if ( $unitsParameter->getUnitlessAllowed() ) {
			$message = null;
			$status = CheckResult::STATUS_COMPLIANCE;
		} else {
			$message = ( new ViolationMessage( 'wbqc-violation-message-units' ) )
				->withEntityId( $snak->getPropertyId(), Role::CONSTRAINT_PROPERTY )
				->withEntityIdList( $unitsParameter->getUnitItemIds(), Role::CONSTRAINT_PARAMETER_VALUE );
			$status = CheckResult::STATUS_VIOLATION;
		}

		return new CheckResult( $context, $constraint, $status, $message );
	}

	/**
	 * Convert $value to standard units.
	 *
	 * @param UnboundedQuantityValue $value
	 * @return UnboundedQuantityValue
	 */
	private function standardize( UnboundedQuantityValue $value ) {
		if ( $this->unitConverter === null ) {
			return $value;
		}

		$standard = $this->unitConverter->toStandardUnits( $value );
		if ( $standard !== null ) {
			return $standard;
		} else {
			return $value;
		}
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		$constraintParameters = $constraint->getConstraintParameters();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parseItemsParameter(
				$constraintParameters,
				$constraint->getConstraintTypeItemId(),
				true
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

}
