<?php

namespace WikibaseQuality\ConstraintReport\Tests\Context;

use LogicException;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\AbstractContext
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class MainSnakContextTest extends \PHPUnit\Framework\TestCase {

	public function testGetSnak() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$context = new MainSnakContext( $entity, $statement );

		$this->assertSame( $statement->getMainSnak(), $context->getSnak() );
	}

	public function testGetEntity() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$context = new MainSnakContext( $entity, $statement );

		$this->assertSame( $entity, $context->getEntity() );
	}

	public function testGetType() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$context = new MainSnakContext( $entity, $statement );

		$this->assertSame( Context::TYPE_STATEMENT, $context->getType() );
	}

	public function testGetSnakRank() {
		$entity = NewItem::withId( 'Q1' )->build();
		$rank = Statement::RANK_DEPRECATED;
		$statement = NewStatement::noValueFor( 'P1' )
			->withRank( $rank )
			->build();
		$context = new MainSnakContext( $entity, $statement );

		$this->assertSame( $rank, $context->getSnakRank() );
	}

	public function testGetSnakStatement() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$context = new MainSnakContext( $entity, $statement );

		$this->assertSame( $statement, $context->getSnakStatement() );
	}

	public function testGetSnakGroup_NonDeprecated() {
		$statement1 = NewStatement::noValueFor( 'P1' )->build();
		$statement2 = NewStatement::noValueFor( 'P1' )->build();
		$statement3 = NewStatement::noValueFor( 'P2' )
			->withDeprecatedRank()
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->andStatement( $statement3 )
			->build();
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( Context::GROUP_NON_DEPRECATED, [] );

		$this->assertSame( [ $statement1->getMainSnak(), $statement2->getMainSnak() ], $snakGroup );
	}

	public function testGetSnakGroup_BestRank() {
		$statement1 = NewStatement::noValueFor( 'P1' )
			->withPreferredRank()
			->build();
		$statement2 = NewStatement::noValueFor( 'P1' )
			->withNormalRank()
			->build();
		$statement3 = NewStatement::noValueFor( 'P2' )
			->withNormalRank()
			->build();
		$statement4 = NewStatement::noValueFor( 'P3' )
			->withDeprecatedRank()
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->andStatement( $statement3 )
			->andStatement( $statement4 )
			->build();
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( Context::GROUP_BEST_RANK );

		$this->assertSame( [ $statement1->getMainSnak(), $statement3->getMainSnak() ], $snakGroup );
	}

	public function testGetSnakGroup_UnknownMode() {
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$entity = NewItem::withId( 'Q1' )->andStatement( $statement )->build();
		$context = new MainSnakContext( $entity, $statement );

		$this->expectException( LogicException::class );
		$context->getSnakGroup( 'unknown-mode' );
	}

	public function testGetSnakGroup_OneSeparator_CustomValues() {
		$mode = Context::GROUP_NON_DEPRECATED; // shouldn’t matter
		$separator = new NumericPropertyId( 'P2' );
		$statement1 = NewStatement::noValueFor( 'P1' )
			->withQualifier( $separator, 'foo' )
			->build();
		$statement2 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator, 'foo' )
			->build();
		$statement3 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator, 'bar' )
			->build();
		$statement4 = NewStatement::someValueFor( 'P1' )
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->andStatement( $statement3 )
			->andStatement( $statement4 )
			->build();
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( $mode, [ $separator ] );

		$expected = [
			$statement1->getMainSnak(),
			$statement2->getMainSnak(),
		];
		$this->assertSame( $expected, $snakGroup );
	}

	public function testGetSnakGroup_OneSeparator_MultipleValues() {
		$mode = Context::GROUP_NON_DEPRECATED; // shouldn’t matter
		$separator = new NumericPropertyId( 'P2' );
		$statement1 = NewStatement::noValueFor( 'P1' )
			->withQualifier( $separator, 'foo' )
			->withQualifier( $separator, 'bar' )
			->build();
		$statement2 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator, 'foo' )
			->withQualifier( $separator, 'bar' )
			->build();
		$statement3 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator, 'bar' )
			->withQualifier( $separator, 'foo' )
			->build();
		$statement4 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator, 'bar' )
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->andStatement( $statement3 )
			->andStatement( $statement4 )
			->build();
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( $mode, [ $separator ] );

		$expected = [
			$statement1->getMainSnak(),
			$statement2->getMainSnak(),
			$statement3->getMainSnak(),
		];
		$this->assertSame( $expected, $snakGroup );
	}

	public function testGetSnakGroup_OneSeparator_UnknownValues() {
		$mode = Context::GROUP_NON_DEPRECATED; // shouldn’t matter
		$separator = new NumericPropertyId( 'P2' );
		$statement1 = NewStatement::noValueFor( 'P1' )
			->build();
		$statement1->getQualifiers()->addSnak( new PropertySomeValueSnak( $separator ) );
		$statement2 = NewStatement::someValueFor( 'P1' )
			->build();
		$statement2->getQualifiers()->addSnak( new PropertySomeValueSnak( $separator ) );
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->build();
		$statement1 = $entity->getStatements()->getIterator()[0]; // NewItem clones statements, we need the final one
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( $mode, [ $separator ] );

		$this->assertSame( [ $statement1->getMainSnak() ], $snakGroup );
	}

	public function testGetSnakGroup_OneSeparator_NoValues() {
		$mode = Context::GROUP_NON_DEPRECATED; // shouldn’t matter
		$separator = new NumericPropertyId( 'P2' );
		$statement1 = NewStatement::noValueFor( 'P1' )
			->build();
		$statement1->getQualifiers()->addSnak( new PropertyNoValueSnak( $separator ) );
		$statement2 = NewStatement::someValueFor( 'P1' )
			->build();
		$statement2->getQualifiers()->addSnak( new PropertyNoValueSnak( $separator ) );
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->build();
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( $mode, [ $separator ] );

		$expected = [
			$statement1->getMainSnak(),
			$statement2->getMainSnak(),
		];
		$this->assertSame( $expected, $snakGroup );
	}

	public function testGetSnakGroup_MultipleSeparators() {
		$mode = Context::GROUP_NON_DEPRECATED; // shouldn’t matter
		$separator1 = new NumericPropertyId( 'P2' );
		$separator2 = new NumericPropertyId( 'P3' );
		$separator3 = new NumericPropertyId( 'P4' );
		$separator4 = new NumericPropertyId( 'P5' );
		$separators = [ $separator1, $separator2, $separator3, $separator4 ];
		$statement1 = NewStatement::noValueFor( 'P1' )
			->withQualifier( $separator1, 'foo' )
			->withQualifier( $separator2, 'bar' )
			->withQualifier( $separator3, 'baz' )
			->build();
		$statement2 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator1, 'foo' )
			->withQualifier( $separator2, 'bar' )
			->withQualifier( $separator3, 'baz' )
			->build();
		$statement3 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator3, 'baz' )
			->withQualifier( $separator2, 'bar' )
			->withQualifier( $separator1, 'foo' )
			->build();
		$statement4 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator1, 'foo' )
			->withQualifier( $separator2, 'bar' )
			->withQualifier( $separator3, 'baz' )
			->withQualifier( $separator4, 'qux' )
			->build();
		$statement5 = NewStatement::someValueFor( 'P1' )
			->withQualifier( $separator1, 'foo' )
			->withQualifier( $separator2, 'baz' )
			->withQualifier( $separator3, 'bar' )
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->andStatement( $statement3 )
			->andStatement( $statement4 )
			->andStatement( $statement5 )
			->build();
		$context = new MainSnakContext( $entity, $statement1 );

		$snakGroup = $context->getSnakGroup( $mode, $separators );

		$expected = [
			$statement1->getMainSnak(),
			$statement2->getMainSnak(),
			$statement3->getMainSnak(),
		];
		$this->assertSame( $expected, $snakGroup );
	}

	public function testGetCursor() {
		$entity = NewItem::withId( 'Q1' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$context = new MainSnakContext( $entity, $statement );

		$cursor = $context->getCursor();

		$this->assertInstanceOf( MainSnakContextCursor::class, $cursor );
	}

}
