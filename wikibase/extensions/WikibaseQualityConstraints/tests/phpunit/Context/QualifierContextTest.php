<?php

namespace WikibaseQuality\ConstraintReport\Tests\Context;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\AbstractContext
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContext
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class QualifierContextTest extends \PHPUnit\Framework\TestCase {

	public function testGetSnak() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$context = new QualifierContext( $entity, $statement, $snak );

		$this->assertSame( $snak, $context->getSnak() );
	}

	public function testGetEntity() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$context = new QualifierContext( $entity, $statement, $snak );

		$this->assertSame( $entity, $context->getEntity() );
	}

	public function testGetType() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$context = new QualifierContext( $entity, $statement, $snak );

		$this->assertSame( Context::TYPE_QUALIFIER, $context->getType() );
	}

	public function testGetSnakRank() {
		$entity = NewItem::withId( 'Q1' )->build();
		$rank = Statement::RANK_DEPRECATED;
		$statement = NewStatement::noValueFor( 'P1' )
			->withRank( $rank )
			->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$context = new QualifierContext( $entity, $statement, $snak );

		$this->assertNull( $context->getSnakRank() );
	}

	public function testGetSnakStatement() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$context = new QualifierContext( $entity, $statement, $snak );

		$this->assertNull( $context->getSnakStatement() );
	}

	/**
	 * @dataProvider provideGroupingModes
	 */
	public function testGetSnakGroup( $groupingMode ) {
		$qualifier1 = new PropertyNoValueSnak( new NumericPropertyId( 'P10' ) );
		$qualifier2 = new PropertySomeValueSnak( new NumericPropertyId( 'P20' ) );
		$statement = new Statement(
			new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
			new SnakList( [ $qualifier1, $qualifier2 ] )
		);
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement )
			->andStatement( NewStatement::someValueFor( 'P2' ) )
			->build();
		$context = new QualifierContext( $entity, $statement, $qualifier1 );

		$snakGroup = $context->getSnakGroup( $groupingMode );

		$this->assertSame( [ $qualifier1, $qualifier2 ], $snakGroup );
	}

	public static function provideGroupingModes() {
		return [
			[ Context::GROUP_NON_DEPRECATED ],
			[ Context::GROUP_BEST_RANK ],
		];
	}

	public function testGetCursor() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$context = new QualifierContext( $entity, $statement, $snak );

		$cursor = $context->getCursor();

		$this->assertInstanceOf( QualifierContextCursor::class, $cursor );
	}

}
