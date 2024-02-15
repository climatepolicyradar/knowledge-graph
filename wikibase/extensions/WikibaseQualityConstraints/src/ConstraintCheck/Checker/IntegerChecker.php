<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use DataValues\DecimalValue;
use DataValues\QuantityValue;
use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author Amir Sarabadani
 * @license GPL-2.0-or-later
 */
class IntegerChecker implements ConstraintChecker {

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

	public function checkConstraint( Context $context, Constraint $constraint ) {
		$snak = $context->getSnak();

		if ( !$snak instanceof PropertyValueSnak ) {
			// nothing to check
			return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
		}

		$violationMessage = $this->checkSnak( $snak );

		return new CheckResult(
			$context,
			$constraint,
			$violationMessage === null ?
				CheckResult::STATUS_COMPLIANCE :
				CheckResult::STATUS_VIOLATION,
			$violationMessage
		);
	}

	/**
	 * @param PropertyValueSnak $snak
	 * @return ViolationMessage|null
	 */
	public function checkSnak( PropertyValueSnak $snak ) {
		$dataValue = $snak->getDataValue();

		if ( $dataValue instanceof DecimalValue ) {
			if ( !$this->isInteger( $dataValue ) ) {
				return $this->getViolationMessage( 'wbqc-violation-message-integer', $snak );
			}
		} elseif ( $dataValue instanceof UnboundedQuantityValue ) {
			if ( !$this->isInteger( $dataValue->getAmount() ) ) {
				return $this->getViolationMessage( 'wbqc-violation-message-integer', $snak );
			} elseif (
				$dataValue instanceof QuantityValue && (
					!$this->isInteger( $dataValue->getLowerBound() ) ||
					!$this->isInteger( $dataValue->getUpperBound() )
				)
			) {
				return $this->getViolationMessage( 'wbqc-violation-message-integer-bounds', $snak );
			}
		}

		return null;
	}

	/**
	 * @param DecimalValue $decimalValue
	 * @return bool
	 */
	private function isInteger( DecimalValue $decimalValue ) {
		return $decimalValue->getTrimmed()->getFractionalPart() === '';
	}

	/**
	 * @param string $messageKey
	 * @param PropertyValueSnak $snak
	 * @return ViolationMessage
	 */
	private function getViolationMessage( $messageKey, PropertyValueSnak $snak ) {
		return ( new ViolationMessage( $messageKey ) )
			->withEntityId( $snak->getPropertyId(), Role::CONSTRAINT_PROPERTY )
			->withDataValue( $snak->getDataValue() );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		// no parameters
		return [];
	}

}
