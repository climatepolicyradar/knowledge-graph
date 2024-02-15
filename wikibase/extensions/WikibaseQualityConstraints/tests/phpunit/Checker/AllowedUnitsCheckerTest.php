<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use DataValues\StringValue;
use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use Wikibase\Lib\Units\InMemoryUnitStorage;
use Wikibase\Lib\Units\UnitConverter;
use Wikibase\Lib\Units\UnitStorage;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\AllowedUnitsChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\AllowedUnitsChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class AllowedUnitsCheckerTest extends \PHPUnit\Framework\TestCase {

	use ConstraintParameters;
	use ResultAssertions;

	private function getAllowedUnitsChecker( UnitStorage $unitStorage = null ) {
		if ( $unitStorage === null ) {
			$unitStorage = new InMemoryUnitStorage( [] );
		}
		return new AllowedUnitsChecker(
			$this->getConstraintParameterParser(),
			new UnitConverter(
				$unitStorage,
				'http://wikibase.example/entity/'
			)
		);
	}

	public function testAllowedUnitsConstraint_UnknownValue() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) );
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testAllowedUnitsConstraint_StringValue() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			new StringValue( '0.25 portion' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-value-needed-of-type' );
	}

	public function testAllowedUnitsConstraint_NoUnit_Allowed() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, '1' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [
			new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
		] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testAllowedUnitsConstraint_NoUnit_NotAllowed() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, '1' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-units' );
	}

	public function testAllowedUnitsConstraint_SomeUnit_Allowed() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q2', 'Q3' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testAllowedUnitsConstraint_SomeUnit_AllowedWithValueConversion() {
		$checker = $this->getAllowedUnitsChecker(
			new InMemoryUnitStorage( [ 'Q2' => [ 'factor' => 1, 'unit' => 'Q3' ] ] )
		);
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q3', 'Q4' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testAllowedUnitsConstraint_SomeUnit_AllowedWithConstraintConversion() {
		$checker = $this->getAllowedUnitsChecker(
			new InMemoryUnitStorage( [ 'Q3' => [ 'factor' => 1, 'unit' => 'Q2' ] ] )
		);
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q3', 'Q4' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testAllowedUnitsConstraint_SomeUnit_AllowedWithDoubleConversion() {
		$checker = $this->getAllowedUnitsChecker(
			new InMemoryUnitStorage( [
				'Q2' => [ 'factor' => 1, 'unit' => 'Q5' ],
				'Q3' => [ 'factor' => 1, 'unit' => 'Q5' ],
			] )
		);
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q3', 'Q4' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testAllowedUnitsConstraint_SomeUnit_NotAllowed() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q3', 'Q4' ] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-units' );
	}

	public function testAllowedUnitsConstraint_SomeUnit_NotAllowed_OrNone() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [
			'Q1', 'Q3', 'Q4',
			new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
		] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-units-or-none' );
	}

	public function testAllowedUnitsConstraint_SomeUnit_NotAllowed_OnlyNone() {
		$checker = $this->getAllowedUnitsChecker();
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			UnboundedQuantityValue::newFromNumber( 0, 'http://wikibase.example/entity/Q2' )
		);
		$context = new FakeSnakContext( $snak );
		$constraint = $this->getConstraintMock( $this->itemsParameter( [
			new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
		] ) );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-units-none' );
	}

	public function testAllowedUnitsConstraint_DeprecatedStatement() {
		$checker = $this->getAllowedUnitsChecker();
		$statement = NewStatement::forProperty( 'P1' )
			->withValue( UnboundedQuantityValue::newFromNumber( 1, 'Q2' ) )
			->withDeprecatedRank()
			->build();
		$constraint = $this->getConstraintMock( $this->itemsParameter( [ 'Q1' ] ) );
		$entity = NewItem::withId( 'Q1' )
			->build();

		$checkResult = $checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		// this constraint is still checked on deprecated statements
		$this->assertViolation( $checkResult, 'wbqc-violation-message-units' );
	}

	public function testCheckConstraintParameters() {
		$checker = $this->getAllowedUnitsChecker();
		$constraint = $this->getConstraintMock( [] );

		$result = $checker->checkConstraintParameters( $constraint );

		$this->assertCount( 1, $result );
	}

	/**
	 * @param string[] $parameters
	 *
	 * @return Constraint
	 */
	private function getConstraintMock( array $parameters ) {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			->willReturn( $parameters );
		$mock->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q21514353' );

		return $mock;
	}

}
