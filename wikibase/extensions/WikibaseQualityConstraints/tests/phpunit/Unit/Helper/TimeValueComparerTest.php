<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Helper;

use DataValues\TimeValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\NowValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TimeValueComparer;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TimeValueComparer
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class TimeValueComparerTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider provideComparisons
	 */
	public function testGetComparison( $expected, TimeValue $lhs, TimeValue $rhs ) {
		$this->assertContains( $expected, [ -1, 0, 1 ], '$expected must be -1, 0, or 1' );
		$timeValueComparer = new TimeValueComparer();
		$actual = $timeValueComparer->getComparison( $lhs, $rhs );
		switch ( $expected ) {
			case -1:
				$this->assertLessThan( 0, $actual );
				break;
			case 0:
				$this->assertSame( 0, $actual );
				break;
			case 1:
				$this->assertGreaterThan( 0, $actual );
				break;
		}
	}

	public static function provideComparisons() {
		$cases = [
			[ -1, self::getTimeValue( 1970 ), self::getTimeValue( 1971 ) ],
			[ 0, self::getTimeValue( 1970 ), self::getTimeValue( 1970 ) ],
			[ 1, self::getTimeValue( 1971 ), self::getTimeValue( 1970 ) ],
			[ -1, self::getTimeValue( 1970, 6, 1 ), self::getTimeValue( 1970, 6, 2 ) ] ,
			[ 0, self::getTimeValue( 1970, 6, 1 ), self::getTimeValue( 1970, 6, 1 ) ] ,
			[ 1, self::getTimeValue( 1970, 6, 1 ), self::getTimeValue( 1970, 5, 31 ) ] ,
		];

		return $cases;
	}

	public function testGetMinimum_first() {
		$timeValue1 = self::getTimeValue( 1991, 9, 1 );
		$timeValue2 = self::getTimeValue( 1991, 9, 2 );
		$timeValueComparer = new TimeValueComparer();

		$minimum = $timeValueComparer->getMinimum( $timeValue1, $timeValue2 );

		$this->assertSame( $timeValue1, $minimum );
	}

	public function testGetMinimum_second() {
		$timeValue1 = self::getTimeValue( 1991, 9, 2 );
		$timeValue2 = self::getTimeValue( 1991, 9, 1 );
		$timeValueComparer = new TimeValueComparer();

		$minimum = $timeValueComparer->getMinimum( $timeValue1, $timeValue2 );

		$this->assertSame( $timeValue2, $minimum );
	}

	public function testIsFutureTime_past() {
		$past = self::getTimeValue( 2012, 10, 29 );
		$timeValueComparer = new TimeValueComparer();

		$this->assertFalse( $timeValueComparer->isFutureTime( $past ) );
	}

	public function testIsFutureTime_now() {
		$timeValueComparer = new TimeValueComparer();

		/*
		 * The first check might sometimes return false,
		 * if it happens right around the end of one second and the beginning of the next,
		 * but in that case the second check should return true.
		 */
		$result = $timeValueComparer->isFutureTime( new NowValue() ) ||
			$timeValueComparer->isFutureTime( new NowValue() );

		$this->assertTrue( $result );
	}

	public function testIsFutureTime_future() {
		$future = self::getTimeValue( 20012, 10, 29 );
		$timeValueComparer = new TimeValueComparer();

		$this->assertTrue( $timeValueComparer->isFutureTime( $future ) );
	}

	/**
	 * @param int $year
	 * @param int $month
	 * @param int $day
	 *
	 * @return TimeValue
	 */
	private static function getTimeValue( $year, $month = 1, $day = 1 ) {
		$yearString = $year > 0 ? "+$year" : "$year";
		$monthString = $month < 10 ? "0$month" : "$month";
		$dayString = $day < 10 ? "0$day" : "$day";
		return new TimeValue(
			"{$yearString}-{$monthString}-{$dayString}T00:00:00Z",
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
	}

}
