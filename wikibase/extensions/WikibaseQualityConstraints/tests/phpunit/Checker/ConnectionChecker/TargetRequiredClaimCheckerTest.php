<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\ConnectionChecker;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\TargetRequiredClaimChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\TargetRequiredClaimChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class TargetRequiredClaimCheckerTest extends \MediaWikiIntegrationTestCase {

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
	 * @var TargetRequiredClaimChecker
	 */
	private $checker;

	protected function setUp(): void {
		parent::setUp();
		$this->lookup = new InMemoryEntityLookup();
		$this->connectionCheckerHelper = new ConnectionCheckerHelper();
		$this->checker = new TargetRequiredClaimChecker(
			$this->lookup,
			$this->getConstraintParameterParser(),
			$this->connectionCheckerHelper
		);
	}

	public function testTargetRequiredClaimConstraintValid() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q5' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P2' )
					->withValue( new ItemId( 'Q42' ) )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = array_merge(
			$this->propertyParameter( 'P2' ),
			$this->itemsParameter( [ 'Q42' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testTargetRequiredClaimConstraintWrongItem() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q5' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P2' )
					->withValue( new ItemId( 'Q42' ) ) // should be 'Q2'
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = array_merge(
			$this->propertyParameter( 'P2' ),
			$this->itemsParameter( [ 'Q2' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-target-required-claim' );
	}

	public function testTargetRequiredClaimConstraintOnlyProperty() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q5' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P2' )
					->withValue( new ItemId( 'Q42' ) )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P2' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testTargetRequiredClaimConstraintOnlyPropertyButDoesNotExist() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q5' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P2' ) // should be 'P3'
					->withValue( new ItemId( 'Q42' ) )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P3' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-target-required-claim' );
	}

	public function testTargetRequiredClaimConstraintWrongDataTypeForItem() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q5' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P2' )
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

		$constraintParameters = $this->propertyParameter( 'P2' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-value-needed-of-type' );
	}

	public function testTargetRequiredClaimConstraintItemDoesNotExist() {
		$entityId = new ItemId( 'Q1' );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P2' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-target-entity-must-exist' );
	}

	public function testTargetRequiredClaimConstraintNoValueSnak() {
		$entityId = new ItemId( 'Q1' );
		$statement = NewStatement::noValueFor( 'P1' )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P2' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testTargetRequiredClaimConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertDeprecation( $checkResult );
	}

	public function testTargetRequiredClaimConstraintDependedEntityIds() {
		$entityId = new ItemId( 'Q1' );
		$otherEntityId = new ItemId( 'Q5' );
		$otherEntity = NewItem::withId( $otherEntityId )
			->andStatement(
				NewStatement::forProperty( 'P2' )
					->withValue( new ItemId( 'Q42' ) )
			)
			->build();
		$this->lookup->addEntity( $otherEntity );
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( $otherEntityId )
			->build();
		$entity = NewItem::withId( $entityId )
			->andStatement( $statement )
			->build();

		$constraintParameters = array_merge(
			$this->propertyParameter( 'P2' ),
			$this->itemsParameter( [ 'Q42' ] )
		);
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
			 ->willReturn( 'Q21503247' );

		return $mock;
	}

}
