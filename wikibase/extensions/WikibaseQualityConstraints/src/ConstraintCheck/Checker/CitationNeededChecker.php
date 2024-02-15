<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

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
class CitationNeededChecker implements ConstraintChecker {

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

	public function checkConstraint( Context $context, Constraint $constraint ) {
		$referenceList = $context->getSnakStatement()->getReferences();

		if ( $referenceList->isEmpty() ) {
			$message = ( new ViolationMessage( 'wbqc-violation-message-citationNeeded' ) )
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
