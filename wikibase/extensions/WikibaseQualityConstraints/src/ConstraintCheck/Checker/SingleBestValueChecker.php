<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use Wikibase\DataModel\Entity\PropertyId;
use Wikibase\DataModel\Statement\Statement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ValueCountCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;

/**
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class SingleBestValueChecker implements ConstraintChecker {

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	/**
	 * @var ValueCountCheckerHelper
	 */
	private $valueCountCheckerHelper;

	public function __construct(
		ConstraintParameterParser $constraintParameterParser
	) {
		$this->constraintParameterParser = $constraintParameterParser;
		$this->valueCountCheckerHelper = new ValueCountCheckerHelper();
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
	 * Checks 'Single best value' constraint.
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

		$separators = $this->constraintParameterParser->parseSeparatorsParameter(
			$constraint->getConstraintParameters()
		);

		$propertyId = $context->getSnak()->getPropertyId();
		$bestRankCount = $this->valueCountCheckerHelper->getPropertyCount(
			$context->getSnakGroup( Context::GROUP_BEST_RANK, $separators ),
			$propertyId
		);

		if ( $bestRankCount > 1 ) {
			$nonDeprecatedCount = $this->valueCountCheckerHelper->getPropertyCount(
				$context->getSnakGroup( Context::GROUP_NON_DEPRECATED ),
				$propertyId
			);
			$message = $this->getViolationMessage(
				$bestRankCount,
				$nonDeprecatedCount,
				$separators,
				$propertyId
			);
			$status = CheckResult::STATUS_VIOLATION;
		} else {
			$message = null;
			$status = CheckResult::STATUS_COMPLIANCE;
		}

		return new CheckResult( $context, $constraint, $status, $message );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		$constraintParameters = $constraint->getConstraintParameters();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parseSeparatorsParameter( $constraintParameters );
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

	/**
	 * @param int $bestRankCount
	 * @param int $nonDeprecatedCount
	 * @param PropertyId[] $separators
	 * @param PropertyId $propertyId
	 * @return ViolationMessage
	 */
	private function getViolationMessage(
		$bestRankCount,
		$nonDeprecatedCount,
		$separators,
		$propertyId
	) {
		if ( $bestRankCount === $nonDeprecatedCount ) {
			if ( $separators === [] ) {
				$messageKey = 'wbqc-violation-message-single-best-value-no-preferred';
			} else {
				$messageKey = 'wbqc-violation-message-single-best-value-no-preferred-separators';
			}
		} else {
			if ( $separators === [] ) {
				$messageKey = 'wbqc-violation-message-single-best-value-multi-preferred';
			} else {
				$messageKey = 'wbqc-violation-message-single-best-value-multi-preferred-separators';
			}
		}

		return ( new ViolationMessage( $messageKey ) )
			->withEntityId( $propertyId )
			->withEntityIdList( $separators );
	}

}
