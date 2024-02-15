<?php

namespace WikibaseQuality\ConstraintReport\Tests\Fake;

use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;

/**
 * Constraint checker implementation that always returns results with a static status.
 *
 * @license GPL-2.0-or-later
 */
class FakeChecker implements ConstraintChecker {

	/**
	 * @var string
	 */
	private $status;

	/**
	 * @param string $status
	 */
	public function __construct( $status = CheckResult::STATUS_TODO ) {
		$this->status = $status;
	}

	public function getSupportedContextTypes() {
		return [
			Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
			Context::TYPE_QUALIFIER => CheckResult::STATUS_COMPLIANCE,
			Context::TYPE_REFERENCE => CheckResult::STATUS_COMPLIANCE,
		];
	}

	public function getDefaultContextTypes() {
		return [
			Context::TYPE_STATEMENT,
			Context::TYPE_QUALIFIER,
			Context::TYPE_REFERENCE,
		];
	}

	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * @see ConstraintChecker::checkConstraint
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		return new CheckResult( $context, $constraint, $this->status );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
	}

}
