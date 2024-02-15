<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Checker;

use Config;
use DataValues\QuantityValue;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\RangeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Role;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class DiffWithinRangeChecker implements ConstraintChecker {

	/**
	 * @var ConstraintParameterParser
	 */
	private $constraintParameterParser;

	/**
	 * @var RangeCheckerHelper
	 */
	private $rangeCheckerHelper;

	/**
	 * @var Config
	 */
	private $config;

	public function __construct(
		ConstraintParameterParser $constraintParameterParser,
		RangeCheckerHelper $rangeCheckerHelper,
		Config $config
	) {
		$this->constraintParameterParser = $constraintParameterParser;
		$this->rangeCheckerHelper = $rangeCheckerHelper;
		$this->config = $config;
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
	 * @param Constraint $constraint
	 *
	 * @throws ConstraintParameterException
	 * @return array [ DataValue|null $min, DataValue|null $max, NumericPropertyId $property ]
	 */
	private function parseConstraintParameters( Constraint $constraint ) {
		list( $min, $max ) = $this->constraintParameterParser->parseQuantityRangeParameter(
			$constraint->getConstraintParameters(),
			$constraint->getConstraintTypeItemId()
		);
		$property = $this->constraintParameterParser->parsePropertyParameter(
			$constraint->getConstraintParameters(),
			$constraint->getConstraintTypeItemId()
		);

		return [ $min, $max, $property ];
	}

	/**
	 * Check whether the endpoints of a range are in years or not.
	 * @param QuantityValue|null $min
	 * @param QuantityValue|null $max
	 *
	 * @return bool
	 */
	private function rangeInYears( $min, $max ) {
		$yearUnit = $this->config->get( 'WBQualityConstraintsYearUnit' );

		if ( $min !== null && $min->getUnit() === $yearUnit ) {
			return true;
		}
		if ( $max !== null && $max->getUnit() === $yearUnit ) {
			return true;
		}

		return false;
	}

	/**
	 * Checks 'Diff within range' constraint.
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

		$snak = $context->getSnak();

		if ( !$snak instanceof PropertyValueSnak ) {
			// nothing to check
			return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
		}

		$minuend = $snak->getDataValue();
		'@phan-var \DataValues\TimeValue|\DataValues\QuantityValue|\DataValues\UnboundedQuantityValue $minuend';

		/** @var NumericPropertyId $property */
		list( $min, $max, $property ) = $this->parseConstraintParameters( $constraint );

		// checks only the first occurrence of the referenced property
		foreach ( $context->getSnakGroup( Context::GROUP_NON_DEPRECATED ) as $otherSnak ) {
			if (
				!$property->equals( $otherSnak->getPropertyId() ) ||
				!$otherSnak instanceof PropertyValueSnak
			) {
				continue;
			}

			$subtrahend = $otherSnak->getDataValue();
			'@phan-var \DataValues\TimeValue|\DataValues\QuantityValue|\DataValues\UnboundedQuantityValue $subtrahend';
			if ( $subtrahend->getType() === $minuend->getType() ) {
				$diff = $this->rangeInYears( $min, $max ) && $minuend->getType() === 'time' ?
					$this->rangeCheckerHelper->getDifferenceInYears( $minuend, $subtrahend ) :
					$this->rangeCheckerHelper->getDifference( $minuend, $subtrahend );

				if ( $this->rangeCheckerHelper->getComparison( $min, $diff ) > 0 ||
					$this->rangeCheckerHelper->getComparison( $diff, $max ) > 0
				) {
					// at least one of $min, $max is set at this point, otherwise there could be no violation
					$openness = $min !== null ? ( $max !== null ? '' : '-rightopen' ) : '-leftopen';
					// possible message keys:
					// wbqc-violation-message-diff-within-range
					// wbqc-violation-message-diff-within-range-leftopen
					// wbqc-violation-message-diff-within-range-rightopen
					$message = ( new ViolationMessage( "wbqc-violation-message-diff-within-range$openness" ) )
						->withEntityId( $context->getSnak()->getPropertyId(), Role::PREDICATE )
						->withDataValue( $minuend, Role::OBJECT )
						->withEntityId( $otherSnak->getPropertyId(), Role::PREDICATE )
						->withDataValue( $subtrahend, Role::OBJECT );
					if ( $min !== null ) {
						$message = $message->withDataValue( $min, Role::OBJECT );
					}
					if ( $max !== null ) {
						$message = $message->withDataValue( $max, Role::OBJECT );
					}
					$status = CheckResult::STATUS_VIOLATION;
				} else {
					$message = null;
					$status = CheckResult::STATUS_COMPLIANCE;
				}
			} else {
				$message = new ViolationMessage( 'wbqc-violation-message-diff-within-range-must-have-equal-types' );
				$status = CheckResult::STATUS_VIOLATION;
			}

			return new CheckResult( $context, $constraint, $status, $message );
		}

		return new CheckResult( $context, $constraint, CheckResult::STATUS_COMPLIANCE );
	}

	public function checkConstraintParameters( Constraint $constraint ) {
		$constraintParameters = $constraint->getConstraintParameters();
		$constraintTypeItemId = $constraint->getConstraintTypeItemId();
		$exceptions = [];
		try {
			$this->constraintParameterParser->parseQuantityRangeParameter(
				$constraintParameters,
				$constraintTypeItemId
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		try {
			$this->constraintParameterParser->parsePropertyParameter(
				$constraintParameters,
				$constraintTypeItemId
			);
		} catch ( ConstraintParameterException $e ) {
			$exceptions[] = $e;
		}
		return $exceptions;
	}

}
