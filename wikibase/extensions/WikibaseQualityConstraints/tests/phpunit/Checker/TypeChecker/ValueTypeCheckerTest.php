<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\TypeChecker;

use DataValues\StringValue;
use NullStatsdDataFactory;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ValueTypeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\DummySparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TypeCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ValueTypeChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ValueTypeCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var InMemoryEntityLookup
	 */
	private $lookup;

	/**
	 * @var ValueTypeChecker
	 */
	private $checker;

	/**
	 * @var NumericPropertyId
	 */
	private $valueTypePropertyId;

	/**
	 * @var string
	 */
	private $subclassPid;

	/**
	 * @var string
	 */
	private $instanceOfId;

	protected function setUp(): void {
		parent::setUp();

		$this->lookup = new InMemoryEntityLookup();
		$this->subclassPid = self::getDefaultConfig()->get( 'WBQualityConstraintsSubclassOfId' );
		$this->instanceOfId = self::getDefaultConfig()->get( 'WBQualityConstraintsInstanceOfId' );
		$this->checker = new ValueTypeChecker(
			$this->lookup,
			$this->getConstraintParameterParser(),
			new TypeCheckerHelper(
				$this->lookup,
				self::getDefaultConfig(),
				new DummySparqlHelper(),
				new NullStatsdDataFactory()
			),
			self::getDefaultConfig()
		);
		$this->valueTypePropertyId = new NumericPropertyId( 'P1234' );
	}

	private function getItemWithSubclassOfStatement( string $itemId, string $statementItemId ) {
		$entity = NewItem::withId( new ItemId( $itemId ) )
			->andStatement(
				NewStatement::forProperty( $this->subclassPid )
					->withValue( new ItemId( $statementItemId ) )
			)
			->build();

		return $entity;
	}

	private function getItemWithInstanceOfStatement( string $itemId, string $statementItemId ) {
		$entity = NewItem::withId( new ItemId( $itemId ) )
			->andStatement(
				NewStatement::forProperty( $this->instanceOfId )
					->withValue( new ItemId( $statementItemId ) )
			)
			->build();

		return $entity;
	}

	public function testValueTypeConstraintInstanceValid() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q1' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintInstanceValidWithIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q2', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q2' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintInstanceValidWithMoreIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q3', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q3' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintSubclassValid() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q4' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintSubclassValidWithIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q5' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintSubclassValidWithMoreIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q6', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q6' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintInstanceOrSubclassValidViaInstance() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q1' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instanceOrSubclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintInstanceOrSubclassValidViaSubclass() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q4' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instanceOrSubclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testValueTypeConstraintInstanceInvalid() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q1' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-valueType-instance' );
	}

	public function testValueTypeConstraintInstanceInvalidWithIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q2', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q2' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-valueType-instance' );
	}

	public function testValueTypeConstraintInstanceInvalidWithMoreIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q3', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q3' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-valueType-instance' );
	}

	public function testValueTypeConstraintSubclassInvalid() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q4' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-valueType-subclass' );
	}

	public function testValueTypeConstraintSubclassInvalidWithIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q5' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-valueType-subclass' );
	}

	public function testValueTypeConstraintSubclassInvalidWithMoreIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q6', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );

		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q6' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-valueType-subclass' );
	}

	public function testValueTypeConstraintWrongType() {
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new StringValue( 'foo bar baz' ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-value-needed-of-type' );
	}

	public function testValueTypeConstraintNonExistingValue() {
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q100' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-value-entity-must-exist' );
	}

	public function testValueTypeConstraintNonExistingRedirectTarget() {
		$snak = new PropertyValueSnak( $this->valueTypePropertyId, new EntityIdValue( new ItemId( 'Q302' ) ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-value-entity-must-exist' );
	}

	public function testValueTypeConstraintNoValueSnak() {
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $snak ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertDeprecation( $checkResult );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$result = $this->checker->checkConstraintParameters( $constraint );

		$this->assertCount( 2, $result );
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
			 ->willReturn( 'Q21510865' );

		return $mock;
	}

}
