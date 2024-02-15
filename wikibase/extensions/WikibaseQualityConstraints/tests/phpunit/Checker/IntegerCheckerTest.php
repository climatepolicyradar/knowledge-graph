<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use DataValues\DecimalValue;
use DataValues\QuantityValue;
use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\Snak;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\IntegerChecker;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\IntegerChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Amir Sarabadani
 * @license GPL-2.0-or-later
 */
class IntegerCheckerTest extends \PHPUnit\Framework\TestCase {

	use ResultAssertions;

	/**
	 * @param Snak $snak
	 * @param string|null $messageKey key of violation message, or null if compliance is expected
	 * @dataProvider provideSnaks
	 */
	public function testIntegerConstraint( Snak $snak, $messageKey ) {
		$checker = new IntegerChecker();
		$constraint = $this->getConstraintMock( [] );

		$checkResult = $checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		if ( $messageKey === null ) {
			$this->assertCompliance( $checkResult );
		} else {
			$this->assertViolation( $checkResult, $messageKey );
		}
	}

	public static function provideSnaks() {
		$p1 = new NumericPropertyId( 'P1' );
		$decimalValue = new DecimalValue( 725.1 );
		$integerValue = new DecimalValue( 7251 );
		$decimalIntegerValue = new DecimalValue( '7251.0' );

		$quantityValueDecimal = new QuantityValue( $decimalValue, '1', $decimalValue, $decimalValue );
		$quantitySnakDecimal = new PropertyValueSnak( $p1, $quantityValueDecimal );

		$quantityValueDecimalWithDecimalBounds = new QuantityValue(
			$decimalValue,
			'1',
			new DecimalValue( 725.13 ),
			new DecimalValue( 725.07 )
		);
		$quantitySnakDecimalWithDecimalBounds = new PropertyValueSnak( $p1, $quantityValueDecimalWithDecimalBounds );

		$quantityValueInteger = new QuantityValue( $integerValue, '1', $integerValue, $integerValue );
		$quantitySnakInteger = new PropertyValueSnak( $p1, $quantityValueInteger );

		$unboundedQuantityValueDecimal = new UnboundedQuantityValue( $decimalValue, '1' );
		$unboundedQuantitySnakDecimal = new PropertyValueSnak( $p1, $unboundedQuantityValueDecimal );

		$unboundedQuantityValueInteger = new UnboundedQuantityValue( $integerValue, '1' );
		$unboundedQuantitySnakInteger = new PropertyValueSnak( $p1, $unboundedQuantityValueInteger );

		$unboundedQuantityValueDecimalInteger = new UnboundedQuantityValue( $decimalIntegerValue, '1' );
		$unboundedQuantitySnakDecimalInteger = new PropertyValueSnak( $p1, $unboundedQuantityValueDecimalInteger );

		$quantityValueIntegerWithDecimalBounds = new QuantityValue(
			$integerValue,
			'1',
			new DecimalValue( 7251.3 ),
			new DecimalValue( 7250.7 )
		);
		$quantitySnakIntegerWithDecimalBounds = new PropertyValueSnak(
			$p1,
			$quantityValueIntegerWithDecimalBounds
		);

		return [
			[ $quantitySnakDecimal, 'wbqc-violation-message-integer' ],
			[ $quantitySnakDecimalWithDecimalBounds, 'wbqc-violation-message-integer' ],
			[ $unboundedQuantitySnakDecimal, 'wbqc-violation-message-integer' ],
			[ $quantitySnakInteger, null ],
			[ $unboundedQuantitySnakInteger, null ],
			[ $unboundedQuantitySnakDecimalInteger, null ],
			[ $quantitySnakIntegerWithDecimalBounds, 'wbqc-violation-message-integer-bounds' ],
		];
	}

	public function testCheckConstraintParameters() {
		$checker = new IntegerChecker();
		$constraint = $this->getConstraintMock( [] );

		$result = $checker->checkConstraintParameters( $constraint );

		$this->assertSame( [], $result );
	}

	/**
	 * @return Constraint
	 */
	private function getConstraintMock() {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			->willReturn( [] );
		$mock->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q52848401' );

		return $mock;
	}

}
