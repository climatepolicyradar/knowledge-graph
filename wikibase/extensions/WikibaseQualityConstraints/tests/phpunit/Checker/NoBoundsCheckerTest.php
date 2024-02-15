<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use DataValues\DecimalValue;
use DataValues\QuantityValue;
use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\Snak;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\NoBoundsChecker;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\NoBoundsChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Amir Sarabadani
 * @license GPL-2.0-or-later
 */
class NoBoundsCheckerTest extends \PHPUnit\Framework\TestCase {

	use ResultAssertions;

	/**
	 * @param Snak $snak
	 * @param string|null $messageKey key of violation message, or null if compliance is expected
	 * @dataProvider provideSnaks
	 */
	public function testNoBoundsConstraint( Snak $snak, $messageKey ) {
		$checker = new NoBoundsChecker();
		$constraint = $this->getConstraintMock( [] );

		$checkResult = $checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );

		if ( $messageKey === null ) {
			$this->assertCompliance( $checkResult );
		} else {
			$this->assertViolation( $checkResult, $messageKey );
		}
	}

	public static function provideSnaks() {
		$decimalValue = new DecimalValue( 7251 );

		$quantityValue = new QuantityValue( $decimalValue, '1', $decimalValue, $decimalValue );
		$quantitySnak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $quantityValue );

		$unboundedQuantityValue = new UnboundedQuantityValue( $decimalValue, '1' );
		$unboundedQuantitySnak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), $unboundedQuantityValue );

		return [
			[ $quantitySnak, 'wbqc-violation-message-noBounds' ],
			[ $unboundedQuantitySnak, null ],
		];
	}

	public function testCheckConstraintParameters() {
		$checker = new NoBoundsChecker();
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
			->willReturn( 'Q51723761' );

		return $mock;
	}

}
