<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use DataValues\QuantityValue;
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
class NoBoundsChecker implements ConstraintChecker {

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

		if ( $snak->getDataValue() instanceof QuantityValue ) {
			$message = ( new ViolationMessage( 'wbqc-violation-message-noBounds' ) )
				->withEntityId( $context->getSnak()->getPropertyId(), Role::CONSTRAINT_PROPERTY );
			return new CheckResult(
				$context,
				$constraint,
				CheckResult::STATUS_VIOLATION,
				$message
			);
		}

		return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		// no parameters
		return [];
	}

}
