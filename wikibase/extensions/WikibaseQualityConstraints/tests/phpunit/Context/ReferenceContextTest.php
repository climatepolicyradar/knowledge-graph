<?php

namespace WikibaseQuality\ConstraintReport\Tests\Context;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Reference;
use Wikibase\DataModel\ReferenceList;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\AbstractContext
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContext
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ReferenceContextTest extends \PHPUnit\Framework\TestCase {

	public function testGetSnak() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$reference = new Reference( [ $snak ] );
		$statement->getReferences()->addReference( $reference );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );

		$this->assertSame( $snak, $context->getSnak() );
	}

	public function testGetEntity() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$reference = new Reference( [ $snak ] );
		$statement->getReferences()->addReference( $reference );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );

		$this->assertSame( $entity, $context->getEntity() );
	}

	public function testGetType() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$reference = new Reference( [ $snak ] );
		$statement->getReferences()->addReference( $reference );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );

		$this->assertSame( Context::TYPE_REFERENCE, $context->getType() );
	}

	public function testGetSnakRank() {
		$entity = NewItem::withId( 'Q1' )->build();
		$rank = Statement::RANK_DEPRECATED;
		$statement = NewStatement::noValueFor( 'P1' )
			->withRank( $rank )
			->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$reference = new Reference( [ $snak ] );
		$statement->getReferences()->addReference( $reference );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );

		$this->assertNull( $context->getSnakRank() );
	}

	public function testGetSnakStatement() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$snak = new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) );
		$reference = new Reference( [ $snak ] );
		$statement->getReferences()->addReference( $reference );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );

		$this->assertNull( $context->getSnakStatement() );
	}

	/**
	 * @dataProvider provideGroupingModes
	 */
	public function testGetSnakGroup( $groupingMode ) {
		$referenceSnak1 = new PropertyNoValueSnak( new NumericPropertyId( 'P100' ) );
		$referenceSnak2 = new PropertySomeValueSnak( new NumericPropertyId( 'P200' ) );
		$referenceSnak3 = new PropertyNoValueSnak( new NumericPropertyId( 'P300' ) );
		$referenceSnak4 = new PropertySomeValueSnak( new NumericPropertyId( 'P400' ) );
		$reference1 = new Reference( [ $referenceSnak1, $referenceSnak2 ] );
		$reference2 = new Reference( [ $referenceSnak3 ] );
		$reference3 = new Reference( [ $referenceSnak4 ] );
		$statement1 = new Statement(
			new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
			/* qualifiers = */ new SnakList( [ $referenceSnak3 ] ),
			new ReferenceList( [ $reference1, $reference2 ] )
		);
		$statement2 = new Statement(
			new PropertySomeValueSnak( new NumericPropertyId( 'P2' ) ),
			null,
			new ReferenceList( [ $reference2, $reference3 ] )
		);
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->build();
		$context = new ReferenceContext( $entity, $statement1, $reference1, $referenceSnak1 );

		$snakGroup = $context->getSnakGroup( $groupingMode );

		$this->assertSame( [ $referenceSnak1, $referenceSnak2 ], $snakGroup );
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
		$reference = new Reference( [ $snak ] );
		$statement->getReferences()->addReference( $reference );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );

		$cursor = $context->getCursor();

		$this->assertInstanceOf( ReferenceContextCursor::class, $cursor );
	}

}
