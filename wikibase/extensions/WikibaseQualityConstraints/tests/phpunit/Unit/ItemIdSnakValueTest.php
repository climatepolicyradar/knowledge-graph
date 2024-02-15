<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit;

use DataValues\StringValue;
use DomainException;
use InvalidArgumentException;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ItemIdSnakValueTest extends \MediaWikiUnitTestCase {

	public function testFromItemId() {
		$itemId = new ItemId( 'Q1' );

		$value = ItemIdSnakValue::fromItemId( $itemId );

		$this->assertTrue( $value->isValue() );
		$this->assertFalse( $value->isSomeValue() );
		$this->assertFalse( $value->isNoValue() );
		$this->assertSame( $itemId, $value->getItemId() );
	}

	public function testSomeValue() {
		$value = ItemIdSnakValue::someValue();

		$this->assertFalse( $value->isValue() );
		$this->assertTrue( $value->isSomeValue() );
		$this->assertFalse( $value->isNoValue() );

		$this->expectException( DomainException::class );
		$value->getItemId();
	}

	public function testNoValue() {
		$value = ItemIdSnakValue::noValue();

		$this->assertFalse( $value->isValue() );
		$this->assertFalse( $value->isSomeValue() );
		$this->assertTrue( $value->isNoValue() );

		$this->expectException( DomainException::class );
		$value->getItemId();
	}

	public function testFromSnak_ItemId() {
		$itemId = new ItemId( 'Q1' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P100' ),
			new EntityIdValue( $itemId )
		);

		$value = ItemIdSnakValue::fromSnak( $snak );

		$this->assertTrue( $value->isValue() );
		$this->assertSame( $itemId, $value->getItemId() );
	}

	public function testFromSnak_PropertyId() {
		$propertyId = new NumericPropertyId( 'P1' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P100' ),
			new EntityIdValue( $propertyId )
		);

		$this->expectException( InvalidArgumentException::class );
		$value = ItemIdSnakValue::fromSnak( $snak );
	}

	public function testFromSnak_String() {
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P100' ),
			new StringValue( 'Q1' )
		);

		$this->expectException( InvalidArgumentException::class );
		$value = ItemIdSnakValue::fromSnak( $snak );
	}

	public function testFromSnak_SomeValue() {
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P100' ) );

		$value = ItemIdSnakValue::fromSnak( $snak );

		$this->assertTrue( $value->isSomeValue() );
	}

	public function testFromSnak_NoValue() {
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P100' ) );

		$value = ItemIdSnakValue::fromSnak( $snak );

		$this->assertTrue( $value->isNoValue() );
	}

	public function testMatchesSnak_ItemId() {
		$itemId = new ItemId( 'Q1' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P100' ),
			new EntityIdValue( $itemId )
		);

		$this->assertTrue( ItemIdSnakValue::fromItemId( $itemId )->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::someValue()->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::noValue()->matchesSnak( $snak ) );
	}

	public function testMatchesSnak_PropertyId() {
		$itemId = new ItemId( 'Q1' );
		$propertyId = new NumericPropertyId( 'P1' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P100' ),
			new EntityIdValue( $propertyId )
		);

		$this->assertFalse( ItemIdSnakValue::fromItemId( $itemId )->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::someValue()->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::noValue()->matchesSnak( $snak ) );
	}

	public function testMatchesSnak_String() {
		$itemId = new ItemId( 'Q1' );
		$propertyId = new NumericPropertyId( 'P1' );
		$snak = new PropertyValueSnak(
			new NumericPropertyId( 'P100' ),
			new StringValue( 'Q1' )
		);

		$this->assertFalse( ItemIdSnakValue::fromItemId( $itemId )->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::someValue()->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::noValue()->matchesSnak( $snak ) );
	}

	public function testMatchesSnak_SomeValue() {
		$itemId = new ItemId( 'Q1' );
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P100' ) );

		$this->assertFalse( ItemIdSnakValue::fromItemId( $itemId )->matchesSnak( $snak ) );
		$this->assertTrue( ItemIdSnakValue::someValue()->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::noValue()->matchesSnak( $snak ) );
	}

	public function testMatchesSnak_NoValue() {
		$itemId = new ItemId( 'Q1' );
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P100' ) );

		$this->assertFalse( ItemIdSnakValue::fromItemId( $itemId )->matchesSnak( $snak ) );
		$this->assertFalse( ItemIdSnakValue::someValue()->matchesSnak( $snak ) );
		$this->assertTrue( ItemIdSnakValue::noValue()->matchesSnak( $snak ) );
	}

}
