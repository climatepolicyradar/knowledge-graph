<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\TypeChecker;

use NullStatsdDataFactory;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\EntityLookup;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementList;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedBool;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\DummySparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\SparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TypeCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TypeCheckerHelper
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class TypeCheckerHelperTest extends \PHPUnit\Framework\TestCase {

	use ConstraintParameters;

	/**
	 * @param EntityLookup|null $entityLookup
	 * @param SparqlHelper|null $sparqlHelper
	 *
	 * @return TypeCheckerHelper
	 */
	private function getHelper(
		EntityLookup $entityLookup = null,
		SparqlHelper $sparqlHelper = null
	) {
		return new TypeCheckerHelper(
			$entityLookup ?: new InMemoryEntityLookup(),
			self::getDefaultConfig(),
			$sparqlHelper ?: new DummySparqlHelper(),
			new NullStatsdDataFactory()
		);
	}

	/**
	 * @param EntityLookup $lookup The backing lookup of the mock.
	 *
	 * @return EntityLookup Expects that getEntity is called
	 * exactly WBQualityConstraintsTypeCheckMaxEntities times.
	 */
	private function getMaxEntitiesLookup( EntityLookup $lookup ) {
		$maxEntities = self::getDefaultConfig()->get( 'WBQualityConstraintsTypeCheckMaxEntities' );

		$spy = $this->createMock( EntityLookup::class );
		$spy->expects( $this->exactly( $maxEntities ) )
			->method( 'getEntity' )
			->willReturnCallback( [ $lookup, 'getEntity' ] );

		return $spy;
	}

	/**
	 * @param boolean $return
	 * @param array|null $arguments
	 *
	 * @return SparqlHelper expects that {@link SparqlHelper::hasType}
	 * is called exactly once and returns $return.
	 */
	private function getSparqlHelper( $return, array $arguments = null ) {
		if ( $arguments === null ) {
			$arguments = [ $this->anything(), $this->anything() ];
		}
		$mock = $this->createMock( SparqlHelper::class );
		$mock->expects( $this->once() )
			->method( 'hasType' )
			->withConsecutive( $arguments )
			->willReturn( new CachedBool( $return, Metadata::blank() ) );
		return $mock;
	}

	/**
	 * Each entity references the Item Id of the second entity
	 * twice, creating a wide structure
	 *
	 * @return EntityLookup $lookup
	 */
	private function getWideEntityStructureLookup(): EntityLookup {
		$subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );
		$entityId = new ItemId( 'Q9' );
		$otherEntityId = new ItemId( 'Q10' );
		$entity = NewItem::withId( $entityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $otherEntityId )
			)
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $otherEntityId )
			)
			->build();
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $entityId )
			)
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $entityId )
			)
			->build();
		$lookup = new InMemoryEntityLookup( $entity, $otherEntity );

		return $lookup;
	}

	public function testHasClassInRelation_Valid() {
		$statement1 = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			new EntityIdValue( new ItemId( 'Q42' ) )
		) );
		$statement2 = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P31' ),
			new EntityIdValue( new ItemId( 'Q1' ) )
		) );
		$statements = new StatementList( $statement1, $statement2 );
		$this->assertTrue( $this->getHelper()->hasClassInRelation( $statements, [ 'P31' ], [ 'Q1' ] )->getBool() );
	}

	public function testHasClassInRelation_Invalid() {
		$statement1 = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			new EntityIdValue( new ItemId( 'Q42' ) )
		) );
		$statement2 = new Statement(
			new PropertyValueSnak( new NumericPropertyId( 'P31' ), new EntityIdValue( new ItemId( 'Q100' ) ) )
		);
		$statements = new StatementList( $statement1, $statement2 );
		$this->assertFalse( $this->getHelper()->hasClassInRelation( $statements, [ 'P31' ], [ 'Q1' ] )->getBool() );
	}

	public function testHasClassInRelation_ValidWithIndirection() {
		$statement1 = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P1' ),
			new EntityIdValue( new ItemId( 'Q42' ) )
		) );
		$statement2 = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P31' ),
			new EntityIdValue( new ItemId( 'Q5' ) )
		) );
		$statements = new StatementList( $statement1, $statement2 );
		$subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );

		$indirectEntity = NewItem::withId( new ItemId( 'Q5' ) )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q4' ) )
			)
			->build();

		$lookup = new InMemoryEntityLookup( $indirectEntity );
		$this->assertTrue( $this->getHelper( $lookup )->hasClassInRelation( $statements, [ 'P31' ], [ 'Q4' ] )->getBool() );
	}

	/**
	 * Test the “instance or subclass of” relation.
	 * The statement list being tested links to Q1 with $firstRelation.
	 * If $secondRelation is not null, then Q1 links to Q2 with $secondRelation.
	 *
	 * @param string $firstRelation 'instance' or 'subclass'
	 * @param string|null $secondRelation 'instance' or 'subclass'
	 * @param string $class item ID serialization
	 * @param bool $expected
	 * @dataProvider provideRelations
	 */
	public function testHasClassInRelation_InstanceOrSubclassOf(
		$firstRelation,
		$secondRelation,
		$class,
		$expected
	) {
		$instanceOfId = self::getDefaultConfig()->get( 'WBQualityConstraintsInstanceOfId' );
		$subclassOfId = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );
		$relationIds = [ $instanceOfId, $subclassOfId ];

		$statements = new StatementList(
			NewStatement::forProperty(
				$firstRelation === 'instance' ?
					$instanceOfId :
					$subclassOfId
				)
				->withValue( new ItemId( 'Q1' ) )
				->build()
		);
		$lookup = new InMemoryEntityLookup();

		if ( $secondRelation !== null ) {
			$q1 = NewItem::withId( 'Q1' )
				->andStatement(
					NewStatement::forProperty(
						$secondRelation === 'instance' ?
							$instanceOfId :
							$subclassOfId
						)
						->withValue( new ItemId( 'Q2' ) )
				)
				->build();
			$lookup->addEntity( $q1 );
		}
		$helper = $this->getHelper( $lookup );

		$result = $helper->hasClassInRelation( $statements, $relationIds, [ $class ] );

		$this->assertSame( $expected, $result->getBool() );
	}

	public function testHasClassInRelation_IgnoresDeprecatedStatement() {
		$statement = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P31' ),
			new EntityIdValue( new ItemId( 'Q1' ) )
		) );
		$statement->setRank( Statement::RANK_DEPRECATED );
		$statements = new StatementList( $statement );
		$this->assertFalse( $this->getHelper()->hasClassInRelation( $statements, [ 'P31' ], [ 'Q1' ] )->getBool() );
	}

	public function testHasClassInRelation_IgnoresDeprecatedSubclassOfStatement() {
		$statement = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P31' ),
			new EntityIdValue( new ItemId( 'Q11' ) ) // Q11 has a deprecated subclass of statement with Q4 as its value
		) );
		$statements = new StatementList( $statement );
		$this->assertFalse( $this->getHelper()->hasClassInRelation( $statements, [ 'P31' ], [ 'Q4' ] )->getBool() );
	}

	public function testHasClassInRelation_IgnoresNonBestSubclassOfStatement() {
		$statement = new Statement( new PropertyValueSnak(
			new NumericPropertyId( 'P31' ),
			new EntityIdValue( new ItemId( 'Q12' ) )
			// Q12 has a normal-rank subclass of statement with Q4 as its value,
			// and a preferred-rank subclass of statement with no value
		) );
		$statements = new StatementList( $statement );
		$this->assertFalse( $this->getHelper()->hasClassInRelation( $statements, [ 'P31' ], [ 'Q4' ] )->getBool() );
	}

	public static function provideRelations() {
		return [
			'direct instance' => [ 'instance', null, 'Q1', true ],
			'direct subclass' => [ 'subclass', null, 'Q1', true ],
			'direct instance of unrelated' => [ 'instance', null, 'Q100', false ],
			'instance of subclass' => [ 'instance', 'subclass', 'Q2', true ],
			'subclass of subclass' => [ 'subclass', 'subclass', 'Q2', true ],
			'subclass of instance' => [ 'subclass', 'instance', 'Q2', false ],
			'instance of subclass of unrelated' => [ 'instance', 'subclass', 'Q200', false ],
		];
	}

	public function testIsSubclassOf_ValidWithIndirection() {
		$subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );
		$entityId = new ItemId( 'Q6' );
		$secondEntityId = new ItemId( 'Q5' );
		$thirdEntityId = new ItemId( 'Q100' ); // entity for testing indirect relation, not part of the lookup itself
		$entity = NewItem::withId( $entityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $secondEntityId )
			)
			->build();
		$secondEntity = NewItem::withId( $secondEntityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $thirdEntityId )
			)
			->build();
		$lookup = new InMemoryEntityLookup( $entity, $secondEntity );
		$helper = $this->getHelper( $lookup );
		$this->assertTrue(
			$helper->isSubclassOfWithSparqlFallback( $entityId, [ 'Q100', 'Q106' ] )->getBool()
		);
	}

	public function testIsSubclassOf_Invalid() {
		$subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );
		$entityId = new ItemId( 'Q6' );
		$entity = NewItem::withId( $entityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q5' ) )
			)
			->build();
		$lookup = new InMemoryEntityLookup( $entity );
		$this->assertFalse(
			$this->getHelper( $lookup )->isSubclassOfWithSparqlFallback( $entityId, [ 'Q200', 'Q201' ] )->getBool()
		);
	}

	public function testIsSubclassOf_Cyclic() {
		$entityId = new ItemId( 'Q7' );
		$subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );
		$otherEntityId = new ItemId( 'Q8' );
		$entity = NewItem::withId( $entityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $otherEntityId )
			)
			->build();
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( $entityId )
			)
			->build();
		$lookup = new InMemoryEntityLookup( $entity, $otherEntity );
		$helper = $this->getHelper( $this->getMaxEntitiesLookup( $lookup ) );
		$this->assertFalse( $helper->isSubclassOfWithSparqlFallback( $entityId, [ 'Q100', 'Q101' ] )->getBool() );
	}

public function testIsSubclassOf_CyclicWide() {
		$lookup = $this->getWideEntityStructureLookup();
		$helper = $this->getHelper( $this->getMaxEntitiesLookup( $lookup ) );
		$this->assertFalse( $helper->isSubclassOfWithSparqlFallback( new ItemId( 'Q9' ), [ 'Q100', 'Q101' ] )->getBool() );
}

	public function testIsSubclassOf_CyclicWideWithSparqlTrue() {
		$lookup = $this->getWideEntityStructureLookup();
		$helper = $this->getHelper( $this->getMaxEntitiesLookup( $lookup ), $this->getSparqlHelper( true ) );
		$this->assertTrue( $helper->isSubclassOfWithSparqlFallback( new ItemId( 'Q9' ), [ 'Q100', 'Q101' ] )->getBool() );
	}

	public function testIsSubclassOf_CyclicWideWithSparqlFalse() {
		$lookup = $this->getWideEntityStructureLookup();
		$helper = $this->getHelper( $this->getMaxEntitiesLookup( $lookup ), $this->getSparqlHelper( false ) );
		$this->assertFalse( $helper->isSubclassOfWithSparqlFallback( new ItemId( 'Q9' ), [ 'Q100', 'Q101' ] )->getBool() );
	}

	public function testIsSubclassOf_Tree() {
		$lookup = new InMemoryEntityLookup();
		$subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );

		$q1 = NewItem::withId( 'Q1' )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q2' ) )
			)
			->build();
		$lookup->addEntity( $q1 );
		$q2 = NewItem::withId( 'Q2' )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q3' ) )
			)
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q5' ) )
			)
			->build();
		$lookup->addEntity( $q2 );
		$q3 = NewItem::withId( 'Q3' )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q4' ) )
			)
			->build();
		$lookup->addEntity( $q3 );
		$q4 = NewItem::withId( 'Q4' )
			->andStatement(
				NewStatement::forProperty( $subclassPid )
					->withValue( new ItemId( 'Q3' ) )
			)
			->build();
		$lookup->addEntity( $q4 );

		$sparqlHelper = $this->getSparqlHelper(
			true,
			[
				$this->identicalTo( 'Q1' ),
				$this->identicalTo( [ 'Q5' ] ),
			]
		);
		$helper = $this->getHelper( $this->getMaxEntitiesLookup( $lookup ), $sparqlHelper );

		$this->assertTrue( $helper->isSubclassOfWithSparqlFallback( new ItemId( 'Q1' ), [ 'Q5' ] )->getBool() );
	}

}
