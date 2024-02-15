<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Checker\ConnectionChecker;

use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementList;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ConnectionCheckerHelperTest extends \MediaWikiUnitTestCase {

	/**
	 * @var StatementList
	 */
	private $statementList;

	/**
	 * @var ConnectionCheckerHelper
	 */
	private $connectionCheckerHelper;

	protected function setUp(): void {
		parent::setUp();
		$this->statementList = new StatementList(
			new Statement( new PropertyValueSnak( new NumericPropertyId( 'P1' ), new EntityIdValue( new ItemId( 'Q1' ) ) ) ),
			new Statement( new PropertyValueSnak( new NumericPropertyId( 'P2' ), new EntityIdValue( new ItemId( 'Q2' ) ) ) ),
			new Statement( new PropertyValueSnak(
				new NumericPropertyId( 'P3' ),
				new EntityIdValue( new NumericPropertyId( 'P10' ) )
			) ),
			new Statement( new PropertySomeValueSnak( new NumericPropertyId( 'P1' ) ) ),
			new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P2' ) ) )
		);
		$this->connectionCheckerHelper = new ConnectionCheckerHelper();
	}

	public function testWithPropertyValid() {
		$statement = $this->connectionCheckerHelper->findStatementWithProperty(
			$this->statementList,
			new NumericPropertyId( 'P1' )
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P1' ), $statement->getPropertyId() );
	}

	public function testWithPropertyInvalid() {
		$statement = $this->connectionCheckerHelper->findStatementWithProperty(
			$this->statementList,
			new NumericPropertyId( 'P100' )
		);

		$this->assertNull( $statement );
	}

	public function testWithPropertyAndEntityIdValueValidItemId() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndEntityIdValue(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			new ItemId( 'Q1' )
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P1' ), $statement->getPropertyId() );
		$this->assertEquals( new ItemId( 'Q1' ), $statement->getMainSnak()->getDataValue()->getEntityId() );
	}

	public function testWithPropertyAndEntityIdValueValidPropertyId() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndEntityIdValue(
			$this->statementList,
			new NumericPropertyId( 'P3' ),
			new NumericPropertyId( 'P10' )
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P3' ), $statement->getPropertyId() );
		$this->assertEquals( new NumericPropertyId( 'P10' ), $statement->getMainSnak()->getDataValue()->getEntityId() );
	}

	public function testWithPropertyAndEntityIdValueInvalidEntityId() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndEntityIdValue(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			new ItemId( 'Q2' )
		);

		$this->assertNull( $statement );
	}

	public function testWithPropertyAndEntityIdValueInvalidProperty() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndEntityIdValue(
			$this->statementList,
			new NumericPropertyId( 'P100' ),
			new ItemId( 'Q1' )
		);

		$this->assertNull( $statement );
	}

	public function testWithPropertyAndItemIdSnakValuesValidItemId() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			[ ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ) ]
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P1' ), $statement->getPropertyId() );
		$this->assertEquals( new ItemId( 'Q1' ), $statement->getMainSnak()->getDataValue()->getEntityId() );
	}

	public function testWithPropertyAndItemIdSnakValuesValidSomeValue() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			[ ItemIdSnakValue::someValue() ]
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P1' ), $statement->getPropertyId() );
		$this->assertSame( 'somevalue', $statement->getMainSnak()->getType() );
	}

	public function testWithPropertyAndItemIdSnakValuesValidNoValue() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P2' ),
			[ ItemIdSnakValue::noValue() ]
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P2' ), $statement->getPropertyId() );
		$this->assertSame( 'novalue', $statement->getMainSnak()->getType() );
	}

	public function testWithPropertyAndItemIdSnakValuesValidMultiple() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			[
				ItemIdSnakValue::noValue(),
				ItemIdSnakValue::fromItemId( new ItemId( 'Q100' ) ),
				ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ),
			]
		);

		$this->assertNotNull( $statement );
		$this->assertEquals( new NumericPropertyId( 'P1' ), $statement->getPropertyId() );
		$this->assertEquals( new ItemId( 'Q1' ), $statement->getMainSnak()->getDataValue()->getEntityId() );
	}

	public function testWithPropertyAndItemIdSnakValuesInvalidItemId() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			[ ItemIdSnakValue::fromItemId( new ItemId( 'Q2' ) ) ]
		);

		$this->assertNull( $statement );
	}

	public function testWithPropertyAndItemIdSnakValuesInvalidSomeValue() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P2' ),
			[ ItemIdSnakValue::someValue() ]
		);

		$this->assertNull( $statement );
	}

	public function testWithPropertyAndItemIdSnakValuesInvalidNoValue() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P1' ),
			[ ItemIdSnakValue::noValue() ]
		);

		$this->assertNull( $statement );
	}

	public function testWithPropertyAndItemIdSnakValuesInvalidProperty() {
		$statement = $this->connectionCheckerHelper->findStatementWithPropertyAndItemIdSnakValues(
			$this->statementList,
			new NumericPropertyId( 'P100' ),
			[ ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ) ]
		);

		$this->assertNull( $statement );
	}

}
