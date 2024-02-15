<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\ValueCountChecker;

use DataValues\StringValue;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Reference;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\UniqueValueChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\DummySparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\SparqlHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;
use WikibaseQuality\ConstraintReport\Tests\SparqlHelperMock;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\UniqueValueChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Olga Bode
 * @license GPL-2.0-or-later
 */
class UniqueValueCheckerTest extends \PHPUnit\Framework\TestCase {

	use ConstraintParameters;
	use SparqlHelperMock;
	use ResultAssertions;

	/**
	 * @var InMemoryEntityLookup
	 */
	private $lookup;

	/**
	 * @var NumericPropertyId
	 */
	private $uniquePropertyId;

	/**
	 * @var UniqueValueChecker
	 */
	private $checker;

	protected function setUp(): void {
		parent::setUp();
		$this->lookup = new InMemoryEntityLookup();
		$this->uniquePropertyId = new NumericPropertyId( 'P31' );
	}

	public function testCheckUniqueValueConstraintInvalid() {
		$entity = NewItem::withId( new ItemId( 'Q6' ) )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( new ItemId( 'Q42' ) )
			)
			->build();
		$this->lookup->addEntity( $entity );
		$statement = new Statement( new PropertyValueSnak( $this->uniquePropertyId, new EntityIdValue( new ItemId( 'Q6' ) ) ) );
		$statement->setGuid( 'Q6$e35707be-4a84-61fe-9b52-623784a316a7' );

		$mock = $this->getSparqlHelperMockFindEntities( $statement, [ new ItemId( 'Q42' ) ] );

		$this->checker = $this->newUniqueValueChecker( $mock );

		$constraint = $this->getConstraintMock( [] );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-unique-value' );
	}

	public function testCheckUniqueValueConstraintInvalidWithPropertyId() {
		$entity = NewItem::withId( new ItemId( 'Q6' ) )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( new ItemId( 'Q42' ) )
			)
			->build();
		$this->lookup->addEntity( $entity );
		$statement = new Statement( new PropertyValueSnak( $this->uniquePropertyId, new EntityIdValue( new ItemId( 'Q6' ) ) ) );
		$statement->setGuid( 'Q6$e35707be-4a84-61fe-9b52-623784a316a7' );

		$mock = $this->getSparqlHelperMockFindEntities( $statement, [ new NumericPropertyId( 'P42' ) ] );

		$this->checker = $this->newUniqueValueChecker( $mock );

		$constraint = $this->getConstraintMock( [] );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-unique-value' );
	}

	public function testCheckUniqueValueConstraintValid() {
		$entity = NewItem::withId( new ItemId( 'Q1' ) )->build();
		$this->lookup->addEntity( $entity );
		$statement = new Statement( new PropertyValueSnak( $this->uniquePropertyId, new EntityIdValue( new ItemId( 'Q1' ) ) ) );
		$statement->setGuid( "Q1$56e6a474-4431-fb24-cc15-1d580e467559" );

		$mock = $this->getSparqlHelperMockFindEntities( $statement, [] );

		$this->checker = $this->newUniqueValueChecker( $mock );

		$entity = $this->lookup->getEntity( new ItemId( 'Q1' ) );

		$constraint = $this->getConstraintMock( [] );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testCheckUniqueValueConstraintInvalidOnQualifier() {
		$entity = NewItem::withId( 'Q6' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$entity->getStatements()->addStatement( $statement );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P7' ), new StringValue( 'O8' ) );
		$sparqlHelper = $this->getSparqlHelperMockFindEntitiesQualifierReference(
			new ItemId( 'Q6' ),
			$snak,
			'qualifier',
			[ new ItemId( 'Q42' ) ]
		);
		$this->checker = $this->newUniqueValueChecker( $sparqlHelper );
		$context = new QualifierContext( $entity, $statement, $snak );
		$constraint = $this->getConstraintMock( [] );

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-unique-value' );
	}

	public function testCheckUniqueValueConstraintValidOnReference() {
		$entity = NewItem::withId( 'Q6' )->build();
		$statement = NewStatement::noValueFor( 'P1' )->build();
		$entity->getStatements()->addStatement( $statement );
		$reference = new Reference();
		$statement->getReferences()->addReference( $reference );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P7' ), new StringValue( 'O8' ) );
		$reference->getSnaks()->addSnak( $snak );
		$sparqlHelper = $this->getSparqlHelperMockFindEntitiesQualifierReference(
			new ItemId( 'Q6' ),
			$snak,
			'reference',
			[]
		);
		$this->checker = $this->newUniqueValueChecker( $sparqlHelper );
		$context = new ReferenceContext( $entity, $statement, $reference, $snak );
		$constraint = $this->getConstraintMock( [] );

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testCheckUniqueValueWithoutSparql() {
		$entity = NewItem::withId( new ItemId( 'Q1' ) )->build();
		$this->lookup->addEntity( $entity );
		$statement = new Statement( new PropertyValueSnak( $this->uniquePropertyId, new EntityIdValue( new ItemId( 'Q1' ) ) ) );
		$statement->setGuid( "Q1$56e6a474-4431-fb24-cc15-1d580e467559" );

		$this->checker = $this->newUniqueValueChecker( new DummySparqlHelper() );

		$constraint = $this->getConstraintMock( [] );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertTodo( $checkResult );
	}

	public function testUniqueValueConstraintDeprecatedStatement() {
		$this->checker = $this->newUniqueValueChecker( new DummySparqlHelper() );
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
		$this->checker = $this->newUniqueValueChecker( new DummySparqlHelper() );
		$constraint = $this->getConstraintMock( [] );

		$result = $this->checker->checkConstraintParameters( $constraint );

		$this->assertSame( [], $result );
	}

	public function testUsesStatementValidQualifiersAsSeparators() {
		$firstQualifier = new NumericPropertyId( 'P10' );
		$secondQualifier = new NumericPropertyId( 'P11' );

		$statement = new Statement(
			new PropertyValueSnak(
				new NumericPropertyId( 'P1' ),
				new StringValue( 'something' )
			),
			new SnakList(
				[
					new PropertyValueSnak( $firstQualifier, new StringValue( 'something' ) ),
					new PropertyValueSnak( $secondQualifier, new StringValue( 'something' ) ),
				]
			)
		);
		$paramParser = $this->getConstraintParameterParserMock( [ $secondQualifier ] );

		$mock = $this->getSparqlHelperMockFindEntities( $statement, [], [ $secondQualifier ] );
		$this->checker = $this->newUniqueValueChecker( $mock, $paramParser );
		$checkResult = $this->checker->checkConstraint(
			new MainSnakContext( new Item( new ItemId( 'Q1' ) ), $statement ),
			$this->getConstraintMock( [] )
		);

		$this->assertCompliance( $checkResult );
	}

	private function newUniqueValueChecker(
		SparqlHelper $sparqlHelper,
		ConstraintParameterParser $paramParser = null
	): UniqueValueChecker {
		$paramParser = $paramParser ?? $this->getConstraintParameterParserMock();
		return new UniqueValueChecker( $sparqlHelper, $paramParser );
	}

	private function getConstraintParameterParserMock( array $separators = [] ): ConstraintParameterParser {
		$paramParser = $this->createMock( ConstraintParameterParser::class );
		$paramParser->method( 'parseSeparatorsParameter' )
			->willReturn( $separators );

		return $paramParser;
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
			 ->willReturn( 'Q21502410' );

		return $mock;
	}

}
