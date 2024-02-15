<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\TypeChecker;

use NullStatsdDataFactory;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\TypeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\DummySparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TypeCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\TypeChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class TypeCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var InMemoryEntityLookup
	 */
	private $lookup;

	/**
	 * @var TypeChecker
	 */
	private $checker;

	/**
	 * @var Snak
	 */
	private $typeSnak;

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
		$this->checker = new TypeChecker(
			$this->getConstraintParameterParser(),
			new TypeCheckerHelper(
				$this->lookup,
				self::getDefaultConfig(),
				new DummySparqlHelper(),
				new NullStatsdDataFactory()
			),
			self::getDefaultConfig()
		);
		$this->typeSnak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), new EntityIdValue( new ItemId( 'Q42' ) ) );
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

	public function testTypeConstraintInstanceValid() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintInstanceValidWithIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q2', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintInstanceValidWithMoreIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q3', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintSubclassValid() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintSubclassValidWithIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );
		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	/**
	 * Given the relation Q6->Q5->Q4->Q100, this tests that Q6->Q100 is still a valid relation
	 */
	public function testTypeConstraintSubclassValidWithMoreIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q6', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintInstanceOrSubclassValidViaInstance() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instanceOrSubclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintInstanceOrSubclassValidViaSubclass() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instanceOrSubclass' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertCompliance( $checkResult );
	}

	public function testTypeConstraintInstanceInvalid() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-instance' );
	}

	/**
	 * Violation should occur since there's no relation between Q2 and Q200
	 */
	public function testTypeConstraintInstanceInvalidWithIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q2', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-instance' );
	}

	public function testTypeConstraintInstanceInvalidWithMoreIndirection() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q3', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-instance' );
	}

	public function testTypeConstraintSubclassInvalid() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );
		$this->lookup->addEntity( $entity );
		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-subclass' );
	}

	public function testTypeConstraintSubclassInvalidWithIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-subclass' );
	}

	public function testTypeConstraintSubclassInvalidWithMoreIndirection() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q6', 'Q5' );
		$secondEntity = $this->getItemWithSubclassOfStatement( 'Q5', 'Q4' );
		$thirdEntity = $this->getItemWithSubclassOfStatement( 'Q4', 'Q100' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $secondEntity );
		$this->lookup->addEntity( $thirdEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'subclass' ),
			$this->classParameter( [ 'Q200', 'Q201' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-subclass' );
	}

	/**
	 * Subclass Cycle: Q7 is a subclass of Q8 and vice versa
	 */
	public function testTypeConstraintSubclassCycle() {
		$entity = $this->getItemWithSubclassOfStatement( 'Q7', 'Q8' );
		$otherEntity = $this->getItemWithSubclassOfStatement( 'Q8', 'Q7' );

		$this->lookup->addEntity( $entity );
		$this->lookup->addEntity( $otherEntity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new FakeSnakContext( $this->typeSnak, $entity ), $constraint );
		$this->assertViolation( $checkResult, 'wbqc-violation-message-type-instance' );
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

	public function testTypeConstraintInstanceValidQualifier() {
		$entity = $this->getItemWithInstanceOfStatement( 'Q1', 'Q100' );
		$this->lookup->addEntity( $entity );

		$constraintParameters = array_merge(
			$this->relationParameter( 'instance' ),
			$this->classParameter( [ 'Q100', 'Q101' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );
		$context = new QualifierContext(
			$entity,
			new Statement( $this->typeSnak ),
			new PropertyNoValueSnak( new NumericPropertyId( 'P2000' ) )
		);

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
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
			 ->willReturn( 'Q21503250' );

		return $mock;
	}

}
