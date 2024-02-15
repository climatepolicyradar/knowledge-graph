<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\RangeChecker;

use DataValues\DecimalValue;
use DataValues\QuantityValue;
use DataValues\TimeValue;
use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\InMemoryDataTypeLookup;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use Wikibase\Lib\Units\CSVUnitStorage;
use Wikibase\Lib\Units\UnitConverter;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\RangeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\RangeCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\RangeChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class RangeCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var TimeValue
	 */
	private $timeValue;

	/**
	 * @var RangeChecker
	 */
	private $checker;

	/**
	 * @var NumericPropertyId
	 */
	private $p1457;

	/**
	 * @var NumericPropertyId
	 */
	private $p2067;

	protected function setUp(): void {
		parent::setUp();
		$this->timeValue = $this->getTimeValue( '1970-01-01' );
		$rangeCheckerHelper = new RangeCheckerHelper(
			self::getDefaultConfig(),
			new UnitConverter( new CSVUnitStorage( __DIR__ . '/units.csv' ), '' )
		);
		$dataTypeLookup = new InMemoryDataTypeLookup();
		$dataTypeLookup->setDataTypeForProperty( new NumericPropertyId( 'P1' ), 'time' );
		$dataTypeLookup->setDataTypeForProperty( new NumericPropertyId( 'P2' ), 'quantity' );
		$this->checker = new RangeChecker(
			$dataTypeLookup,
			$this->getConstraintParameterParser(),
			$rangeCheckerHelper
		);
		$this->p1457 = new NumericPropertyId( 'P1457' );
		$this->p2067 = new NumericPropertyId( 'P2067' );
	}

	public function testRangeConstraintWithinRange() {
		$value = new DecimalValue( 3.1415926536 );
		$snak = new PropertyValueSnak( $this->p1457, $this->getQuantity( $value ) );
		$constraintParameters = $this->rangeParameter( 'quantity', 0, 10 );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintTooSmall() {
		$value = new DecimalValue( 42 );
		$snak = new PropertyValueSnak( $this->p1457, $this->getQuantity( $value ) );
		$constraintParameters = $this->rangeParameter( 'quantity', 100, 1000 );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-quantity-closed' );
	}

	public function testRangeConstraintTooBig() {
		$value = new DecimalValue( 3.141592 );
		$snak = new PropertyValueSnak( $this->p1457, $this->getQuantity( $value ) );
		$constraintParameters = $this->rangeParameter( 'quantity', 0, 1 );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-quantity-closed' );
	}

	public function testRangeConstraintGWithinRange() {
		$value = new DecimalValue( 500.0 );
		$snak = new PropertyValueSnak( $this->p2067, $this->getQuantity( $value, 'g' ) );
		$constraint = $this->getConstraintMock( $this->rangeParameter( 'quantity', 0, 1 ) );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintGTooBig() {
		$value = new DecimalValue( 2000.0 );
		$snak = new PropertyValueSnak( $this->p2067, $this->getQuantity( $value, 'g' ) );
		$constraint = $this->getConstraintMock( $this->rangeParameter( 'quantity', 0, 1 ) );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-quantity-closed' );
	}

	public function testRangeConstraintTimeWithinRange() {
		$snak = new PropertyValueSnak( $this->p1457, $this->timeValue );
		$constraintParameters = $this->rangeParameter( 'date', '1960', '1980' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
		$this->assertNull( $checkResult->getMetadata()->getDependencyMetadata()->getFutureTime() );
	}

	public function testRangeConstraintTimeWithinRangeToNow() {
		$snak = new PropertyValueSnak( $this->p1457, $this->timeValue );
		$constraintParameters = $this->rangeParameter( 'date', '1960', 'now' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
		$this->assertNull( $checkResult->getMetadata()->getDependencyMetadata()->getFutureTime() );
	}

	public function testRangeConstraintTimeWithinYearRange() {
		$snak = new PropertyValueSnak( $this->p1457, $this->timeValue );
		$constraintParameters = $this->rangeParameter( 'date', '1960', '1980' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintTimeWithinYearMonthDayRange() {
		$snak = new PropertyValueSnak( $this->p1457, $this->timeValue );
		$constraintParameters = $this->rangeParameter( 'date', '1969-12-31', '1970-01-02' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintTimeTooEarly() {
		$snak = new PropertyValueSnak( $this->p1457, $this->timeValue );
		$constraintParameters = $this->rangeParameter( 'date', '1975', '1980' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-closed' );
	}

	public function testRangeConstraintTimeTooLate() {
		$snak = new PropertyValueSnak( $this->p1457, $this->timeValue );
		$constraintParameters = $this->rangeParameter( 'date', '1960', '1965' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-closed' );
	}

	public function testRangeConstraintNoValueSnak() {
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) );
		$constraintParameters = [];
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintLeftOpenWithinRange() {
		$snak = new PropertyValueSnak( $this->p1457, UnboundedQuantityValue::newFromNumber( -10 ) );
		$constraintParameters = $this->rangeParameter( 'quantity', null, 0 );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintLeftOpenTooSmall() {
		$snak = new PropertyValueSnak( $this->p1457, UnboundedQuantityValue::newFromNumber( 10 ) );
		$constraintParameters = $this->rangeParameter( 'quantity', null, 0 );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-quantity-leftopen' );
	}

	public function testRangeConstraintRightOpenWithinRange() {
		$snak = new PropertyValueSnak( $this->p1457, UnboundedQuantityValue::newFromNumber( 10 ) );
		$constraintParameters = $this->rangeParameter( 'quantity', 0, null );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintRightOpenTooSmall() {
		$snak = new PropertyValueSnak( $this->p1457, UnboundedQuantityValue::newFromNumber( -10 ) );
		$constraintParameters = $this->rangeParameter( 'quantity', 0, null );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-quantity-rightopen' );
	}

	public function testRangeConstraintLeftOpenTimeWithinRange() {
		$nineteenFourtyNine = $this->getTimeValue( '1949-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $nineteenFourtyNine );
		$constraintParameters = $this->rangeParameter( 'time', null, $this->timeValue );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintLeftOpenTimeTooLate() {
		$nineteenEightyFour = $this->getTimeValue( '1984-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $nineteenEightyFour );
		$constraintParameters = $this->rangeParameter( 'time', null, $this->timeValue );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-leftopen' );
	}

	public function testRangeConstraintRightOpenTimeWithinRange() {
		$nineteenEightyFour = $this->getTimeValue( '1984-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $nineteenEightyFour );
		$constraintParameters = $this->rangeParameter( 'time', $this->timeValue, null );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testRangeConstraintRightOpenTimeTooEarly() {
		$nineteenFourtyNine = $this->getTimeValue( '1949-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $nineteenFourtyNine );
		$constraintParameters = $this->rangeParameter( 'time', $this->timeValue, null );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-rightopen' );
	}

	public function testRangeConstraintOutsideClosedRangeToNow() {
		$misspelledNineteenEightyFour = $this->getTimeValue( '19984-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $misspelledNineteenEightyFour );
		$constraintParameters = $this->rangeParameter( 'time', $this->timeValue, 'now' );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-closed-rightnow' );
		$futureTime = $checkResult->getMetadata()->getDependencyMetadata()->getFutureTime();
		$this->assertNotNull( $futureTime );
		$this->assertSame( $misspelledNineteenEightyFour->getTime(), $futureTime->getTime() );
	}

	public function testRangeConstraintOutsideOpenRangeToNow() {
		$misspelledNineteenEightyFour = $this->getTimeValue( '19984-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $misspelledNineteenEightyFour );
		$constraintParameters = $this->rangeParameter( 'time', null, 'now' );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-leftopen-rightnow' );
		$futureTime = $checkResult->getMetadata()->getDependencyMetadata()->getFutureTime();
		$this->assertNotNull( $futureTime );
		$this->assertSame( $misspelledNineteenEightyFour->getTime(), $futureTime->getTime() );
	}

	public function testRangeConstraintOutsideClosedRangeFromNow() {
		$farFuture = $this->getTimeValue( '19984-01-01' );
		$nineteenEightyFour = $this->getTimeValue( '1984-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $nineteenEightyFour );
		$constraintParameters = $this->rangeParameter( 'time', 'now', $farFuture );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-closed-leftnow' );
		$this->assertNull( $checkResult->getMetadata()->getDependencyMetadata()->getFutureTime() );
	}

	public function testRangeConstraintOutsideOpenRangeFromNow() {
		$nineteenEightyFour = $this->getTimeValue( '1984-01-01' );
		$snak = new PropertyValueSnak( $this->p1457, $nineteenEightyFour );
		$constraintParameters = $this->rangeParameter( 'time', 'now', null );

		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-range-time-rightopen-leftnow' );
		$this->assertNull( $checkResult->getMetadata()->getDependencyMetadata()->getFutureTime() );
	}

	public function testRangeConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertDeprecation( $checkResult );
	}

	/**
	 * @dataProvider providePropertyIds
	 */
	public function testCheckConstraintParameters( $propertyId ) {
		$constraint = $this->getConstraintMock( [], $propertyId );

		$result = $this->checker->checkConstraintParameters( $constraint );

		$this->assertCount( 1, $result );
	}

	public static function providePropertyIds() {
		return [ [ 'P1' ], [ 'P2' ] ];
	}

	/**
	 * @param string[] $parameters
	 * @param string $propertyId
	 *
	 * @return Constraint
	 */
	private function getConstraintMock( array $parameters, $propertyId = 'P1' ) {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			 ->willReturn( $parameters );
		$mock->method( 'getConstraintTypeItemId' )
			 ->willReturn( 'Q21510860' );
		$mock->method( 'getPropertyId' )
			 ->willReturn( new NumericPropertyId( $propertyId ) );

		return $mock;
	}

	private function getQuantity( $value, $unit = '1' ): QuantityValue {
		return new QuantityValue( $value, $unit, $value, $value );
	}

	private function getTimeValue( $date ): TimeValue {
		return new TimeValue(
			"+{$date}T00:00:00Z",
			0,
			0,
			0,
			TimeValue::PRECISION_DAY,
			TimeValue::CALENDAR_GREGORIAN
		);
	}

}
