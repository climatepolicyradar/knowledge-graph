<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\ConnectionChecker;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Entity\Property;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\InverseChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\InverseChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class InverseCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var InMemoryEntityLookup
	 */
	private $lookup;

	/**
	 * @var ConnectionCheckerHelper
	 */
	private $connectionCheckerHelper;

	/**
	 * @var InverseChecker
	 */
	private $checker;

	protected function setUp(): void {
		parent::setUp();
		$this->lookup = new InMemoryEntityLookup();
		$this->connectionCheckerHelper = new ConnectionCheckerHelper();
		$this->checker = new InverseChecker(
			$this->lookup,
			$this->getConstraintParameterParser(),
			$this->connectionCheckerHelper
		);
	}

	public function testInverseConstraintValid() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q7' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( $entityId )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P1' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testInverseConstraintOnProperty() {
		$entityId = new NumericPropertyId( 'P1' );
		$otherEntityId = new NumericPropertyId( 'P2' );
		$otherEntity = new Property( $otherEntityId, null, 'wikibase-property' );
		$otherEntity->getStatements()->addStatement(
			NewStatement::forProperty( 'P3' )
				->withValue( $entityId )
				->build()
		);
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P3' )
			->withValue( $otherEntityId )
			->build();
		$entity = new Property( $entityId, null, 'wikibase-property' );
		$entity->getStatements()->addStatement(
			$statement
		);

		$constraintParameters = $this->propertyParameter( 'P3' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testInverseConstraintWrongItem() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q7' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( new ItemId( 'Q2' ) ) // should be $entityId
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P1' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-inverse' );
	}

	public function testInverseConstraintWrongDataTypeForItem() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q7' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( $entityId )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId->getSerialization() ) // should be $otherEntityId
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P1' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-value-needed-of-type' );
	}

	public function testInverseConstraintItemDoesNotExist() {
		$entityId = new ItemId( 'Q1' );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P1' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-target-entity-must-exist' );
	}

	public function testInverseConstraintNoValueSnak() {
		$entityId = new ItemId( 'Q1' );
		$statement = NewStatement::noValueFor( 'P188' )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P1' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testInverseConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertDeprecation( $checkResult );
	}

	public function testInverseConstraintDependedEntityIds() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q7' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( $entityId )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P1' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$dependencyMetadata = $checkResult->getMetadata()->getDependencyMetadata();
		$this->assertSame( [ $otherEntityId ], $dependencyMetadata->getEntityIds() );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$result = $this->checker->checkConstraintParameters( $constraint );

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
			 ->willReturn( 'Q21510855' );

		return $mock;
	}

}
