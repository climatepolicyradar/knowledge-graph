<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Statement\Statement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class QualifiersChecker implements ConstraintChecker {

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	/**
	 * @param ConstraintParameterParser $constraintParameterParser
	 */
	public function __construct(
		ConstraintParameterParser $constraintParameterParser
	) {
		$this->constraintParameterParser = $constraintParameterParser;
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
		return [
			Context::TYPE_STATEMENT,
		];
	}

	/** @codeCoverageIgnore This method is purely declarative. */
	public function getSupportedEntityTypes() {
		return self::ALL_ENTITY_TYPES_SUPPORTED;
	}

	/**
	 * Checks 'Qualifiers' constraint.
	 *
	 * @param Context $context
	 * @param Constraint $constraint
	 *
	 * @throws ConstraintParameterException
	 * @return CheckResult
	 */
	public function checkConstraint( Context $context, Constraint $constraint ) {
		if ( $context->getSnakRank() === Statement::RANK_DEPRECATED ) {
			return new CheckResult( $context, $constraint, CheckResult::STATUS_DEPRECATED );
		}

		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();

		$properties = $this->constraintParameterParser->parsePropertiesParameter(
			$constraintParameters,
			$constraintTypeItemId
		);

		$message = null;
		$status = CheckResult::STATUS_COMPLIANCE;

		/** @var Snak $qualifier */
		foreach ( $context->getSnakStatement()->getQualifiers() as $qualifier ) {
			$allowedQualifier = false;
			foreach ( $properties as $property ) {
				if ( $qualifier->getPropertyId()->equals( $property ) ) {
					$allowedQualifier = true;
					break;
				}
			}
			if ( !$allowedQualifier ) {
				if ( empty( $properties ) || $properties === [ '' ] ) {
					$message = ( new ViolationMessage( 'wbqc-violation-message-no-qualifiers' ) )
						->withEntityId( $context->getSnak()->getPropertyId(), Role::CONSTRAINT_PROPERTY );
				} else {
					$message = ( new ViolationMessage( 'wbqc-violation-message-qualifiers' ) )
						->withEntityId( $context->getSnak()->getPropertyId(), Role::CONSTRAINT_PROPERTY )
						->withEntityId( $qualifier->getPropertyId(), Role::QUALIFIER_PREDICATE )
						->withEntityIdList( $properties, Role::QUALIFIER_PREDICATE );
				}
				$status = CheckResult::STATUS_VIOLATION;
				break;
			}
		}

		return new CheckResult( $context, $constraint, $status, $message );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parsePropertiesParameter(
				$constraintParameters,
				$constraintTypeItemId
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

}
