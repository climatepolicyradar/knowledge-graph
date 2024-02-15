<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Config;
use DataValues\DataValue;
use DataValues\QuantityValue;
use DataValues\TimeValue;
use DataValues\TimeValueCalculator;
use DataValues\UnboundedQuantityValue;
use InvalidArgumentException;
use ValueParsers\IsoTimestampParser;
use ValueParsers\ValueParser;
use Wikibase\Lib\Units\UnitConverter;

/**
 * Class for helper functions for range checkers.
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class RangeCheckerHelper {

	/**
	 * @var Config
	 */
	private $config;

	/**
	 * @var ValueParser
	 */
	private $timeParser;

	/**
	 * @var TimeValueCalculator
	 */
	private $timeCalculator;

	/**
	 * @var TimeValueComparer
	 */
	private $timeValueComparer;

	/**
	 * @var UnitConverter|null
	 */
	private $unitConverter;

	public function __construct(
		Config $config,
		UnitConverter $unitConverter = null
	) {
		$this->config = $config;
		$this->timeParser = new IsoTimestampParser();
		$this->timeCalculator = new TimeValueCalculator();
		$this->timeValueComparer = new TimeValueComparer( $this->timeCalculator );
		$this->unitConverter = $unitConverter;
	}

	/**
	 * @param UnboundedQuantityValue $value
	 * @return UnboundedQuantityValue $value converted to standard units if possible, otherwise unchanged $value.
	 */
	private function standardize( UnboundedQuantityValue $value ) {
		if ( $this->unitConverter !== null ) {
			$standard = $this->unitConverter->toStandardUnits( $value );
			if ( $standard !== null ) {
				return $standard;
			} else {
				return $value;
			}
		} else {
			return $value;
		}
	}

	/**
	 * Compare two values.
	 * If one of them is null, return 0 (equal).
	 *
	 * @param DataValue|null $lhs left-hand side
	 * @param DataValue|null $rhs right-hand side
	 *
	 * @throws InvalidArgumentException if the values do not both have the same, supported data value type
	 * @return integer An integer less than, equal to, or greater than zero
	 *                 when $lhs is respectively less than, equal to, or greater than $rhs.
	 *                 (In other words, just like the “spaceship” operator <=>.)
	 */
	public function getComparison( DataValue $lhs = null, DataValue $rhs = null ) {
		if ( $lhs === null || $rhs === null ) {
			return 0;
		}

		if ( $lhs->getType() !== $rhs->getType() ) {
			throw new InvalidArgumentException( 'Different data value types' );
		}

		switch ( $lhs->getType() ) {
			case 'time':
				/** @var TimeValue $lhs */
				/** @var TimeValue $rhs */
				'@phan-var TimeValue $lhs';
				'@phan-var TimeValue $rhs';
				return $this->timeValueComparer->getComparison( $lhs, $rhs );
			case 'quantity':
				/** @var QuantityValue|UnboundedQuantityValue $lhs */
				/** @var QuantityValue|UnboundedQuantityValue $rhs */
				'@phan-var QuantityValue|UnboundedQuantityValue $lhs';
				'@phan-var QuantityValue|UnboundedQuantityValue $rhs';
				$lhsStandard = $this->standardize( $lhs );
				$rhsStandard = $this->standardize( $rhs );
				return $lhsStandard->getAmount()->compare( $rhsStandard->getAmount() );
		}

		throw new InvalidArgumentException( 'Unsupported data value type' );
	}

	/**
	 * Computes $minuend - $subtrahend, in a format depending on the data type.
	 * For time values, the difference is in seconds;
	 * for quantity values, the difference is the numerical difference between the quantities,
	 * after attempting normalization of each side.
	 *
	 * @param TimeValue|QuantityValue|UnboundedQuantityValue $minuend
	 * @param TimeValue|QuantityValue|UnboundedQuantityValue $subtrahend
	 *
	 * @throws InvalidArgumentException if the values do not both have the same, supported data value type
	 * @return UnboundedQuantityValue
	 */
	public function getDifference( DataValue $minuend, DataValue $subtrahend ) {
		if ( $minuend->getType() === 'time' && $subtrahend->getType() === 'time' ) {
			$minuendSeconds = $this->timeCalculator->getTimestamp( $minuend );
			$subtrahendSeconds = $this->timeCalculator->getTimestamp( $subtrahend );
			return UnboundedQuantityValue::newFromNumber(
				$minuendSeconds - $subtrahendSeconds,
				$this->config->get( 'WBQualityConstraintsSecondUnit' )
			);
		}
		if ( $minuend->getType() === 'quantity' && $subtrahend->getType() === 'quantity' ) {
			$minuendStandard = $this->standardize( $minuend );
			$subtrahendStandard = $this->standardize( $subtrahend );
			$minuendValue = $minuendStandard->getAmount()->getValueFloat();
			$subtrahendValue = $subtrahendStandard->getAmount()->getValueFloat();
			$diff = $minuendValue - $subtrahendValue;
			// we don’t check whether both quantities have the same standard unit –
			// that’s the job of a different constraint type, Units (T164372)
			return UnboundedQuantityValue::newFromNumber( $diff, $minuendStandard->getUnit() );
		}

		throw new InvalidArgumentException( 'Unsupported or different data value types' );
	}

	public function getDifferenceInYears( TimeValue $minuend, TimeValue $subtrahend ) {
		if ( !preg_match( '/^([-+]\d{1,16})-(.*)$/', $minuend->getTime(), $minuendMatches ) ||
			!preg_match( '/^([-+]\d{1,16})-(.*)$/', $subtrahend->getTime(), $subtrahendMatches )
		) {
			throw new InvalidArgumentException( 'TimeValue::getTime() did not match expected format' );
		}
		$minuendYear = (float)$minuendMatches[1];
		$subtrahendYear = (float)$subtrahendMatches[1];
		$minuendRest = $minuendMatches[2];
		$subtrahendRest = $subtrahendMatches[2];

		// calculate difference of years
		$diff = $minuendYear - $subtrahendYear;
		if ( $minuendYear > 0.0 && $subtrahendYear < 0.0 ) {
			$diff -= 1.0; // there is no year 0, remove it from difference
		} elseif ( $minuendYear < 0.0 && $subtrahendYear > 0.0 ) {
			$diff -= -1.0; // there is no year 0, remove it from negative difference
		}

		// adjust for date within year by parsing the month-day part within the same year
		$minuendDateValue = $this->timeParser->parse( '+0000000000001970-' . $minuendRest );
		$subtrahendDateValue = $this->timeParser->parse( '+0000000000001970-' . $subtrahendRest );
		$minuendDateSeconds = $this->timeCalculator->getTimestamp( $minuendDateValue );
		$subtrahendDateSeconds = $this->timeCalculator->getTimestamp( $subtrahendDateValue );
		if ( $minuendDateSeconds < $subtrahendDateSeconds ) {
			// difference in the last year is actually less than one full year
			// e. g. 1975-03-01 - 1974-09-01 is just six months
			// (we don’t need sub-year precision in the difference, adjusting by 0.5 is enough)
			$diff -= 0.5;
		} elseif ( $minuendDateSeconds > $subtrahendDateSeconds ) {
			// difference in the last year is actually more than one full year
			// e. g. 1975-09-01 - 1974-03-01 is 18 months
			// (we don’t need sub-year precision in the difference, adjusting by 0.5 is enough)
			$diff += 0.5;
		}

		$unit = $this->config->get( 'WBQualityConstraintsYearUnit' );
		return UnboundedQuantityValue::newFromNumber( $diff, $unit );
	}

	public function isFutureTime( TimeValue $timeValue ) {
		return $this->timeValueComparer->isFutureTime( $timeValue );
	}

}
