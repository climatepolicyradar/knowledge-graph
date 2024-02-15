<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use DataValues\TimeValue;
use DataValues\TimeValueCalculator;

/**
 *
 * @license GPL-2.0-or-later
 */
class TimeValueComparer {

	/**
	 * @var TimeValueCalculator
	 */
	private $timeValueCalculator;

	public function __construct( TimeValueCalculator $timeValueCalculator = null ) {
		$this->timeValueCalculator = $timeValueCalculator ?: new TimeValueCalculator();
	}

	public function getComparison( TimeValue $lhs, TimeValue $rhs ) {
		$lhsTimestamp = $this->timeValueCalculator->getTimestamp( $lhs );
		$rhsTimestamp = $this->timeValueCalculator->getTimestamp( $rhs );

		if ( $lhsTimestamp === $rhsTimestamp ) {
			return 0;
		}

		return $lhsTimestamp < $rhsTimestamp ? -1 : 1;
	}

	public function getMinimum( TimeValue $timeValue1, TimeValue $timeValue2 ) {
		return $this->getComparison( $timeValue1, $timeValue2 ) <= 0 ? $timeValue1 : $timeValue2;
	}

	public function isFutureTime( TimeValue $timeValue ) {
		return $this->getComparison( $timeValue, new NowValue() ) >= 0;
	}

}
