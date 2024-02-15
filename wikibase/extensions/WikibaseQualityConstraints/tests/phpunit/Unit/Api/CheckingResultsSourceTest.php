<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Api;

use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Statement\Statement;
use WikibaseQuality\ConstraintReport\Api\CheckingResultsSource;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;

/**
 * @covers WikibaseQuality\ConstraintReport\Api\CheckingResultsSource
 *
 * @license GPL-2.0-or-later
 */
class CheckingResultsSourceTest extends \MediaWikiUnitTestCase {

	private const NONEXISTENT_ITEM = 'Q99';
	private const NONEXISTENT_CLAIM = 'Q99$dfb32791-ffd5-4420-a1d9-2bc2a0775968';

	private function getResultsSource(
		DelegatingConstraintChecker $delegatingConstraintChecker = null
	) {
		if ( $delegatingConstraintChecker === null ) {
			$delegatingConstraintChecker = $this->createMock(
				DelegatingConstraintChecker::class
			);
		}

		return new CheckingResultsSource(
			$delegatingConstraintChecker
		);
	}

	public function testGetResults() {
		$q1 = new ItemId( 'Q1' );
		$q2 = new ItemId( 'Q2' );
		$s1 = 'Q3$7f6d761c-bad5-47b6-a335-89635f102771';
		$s2 = 'Q4$41dcb5ec-2ca5-4cfa-822b-a602038fc99f';
		$constraintIds = [ 'P1$47681880-d5f5-417d-96c3-570d6e94d234' ];
		$mock = $this->getMockBuilder( DelegatingConstraintChecker::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'checkAgainstConstraintsOnEntityId', 'checkAgainstConstraintsOnClaimId' ] );
		$delegatingConstraintChecker = $mock->getMock();
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnEntityId' )
			->withConsecutive(
				[ $q1, $constraintIds, $this->callback( 'is_callable' ) ],
				[ $q2, $constraintIds, $this->callback( 'is_callable' ) ]
			)
			->willReturnCallback( function ( $entityId ) {
				return [ new CheckResult(
					new MainSnakContext(
						new Item( $entityId ),
						new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
					),
					new Constraint(
						'P1$47681880-d5f5-417d-96c3-570d6e94d234',
						new NumericPropertyId( 'P1' ),
						'Q1',
						[]
					)
				) ];
			} );
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnClaimId' )
			->withConsecutive(
				[ $s1, $constraintIds, $this->callback( 'is_callable' ) ],
				[ $s2, $constraintIds, $this->callback( 'is_callable' ) ]
			)
			->willReturnCallback( function ( $claimId ) {
				$entityId = new ItemId( substr( $claimId, 0, 2 ) );
				return [ new CheckResult(
					new MainSnakContext(
						new Item( $entityId ),
						new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
					),
					new Constraint(
						'P1$47681880-d5f5-417d-96c3-570d6e94d234',
						new NumericPropertyId( 'P1' ),
						'Q1',
						[]
					)
				) ];
			} );

		$results = $this->getResultsSource( $delegatingConstraintChecker )->getResults(
			[ $q1, $q2 ],
			[ $s1, $s2 ],
			$constraintIds,
			[ CheckResult::STATUS_TODO ]
		)->getArray();

		$this->assertCount( 4, $results );
		$this->assertCount( 4, array_unique( array_map( function ( CheckResult $result ) {
			return $result->getContextCursor()->getEntityId();
		}, $results ) ) );
		foreach ( $results as $result ) {
			$this->assertSame( Context::TYPE_STATEMENT, $result->getContextCursor()->getType() );
			$this->assertSame( 'P1', $result->getContextCursor()->getSnakPropertyId() );
		}
	}

	public function testGetResults_Empty() {
		$mock = $this->getMockBuilder( DelegatingConstraintChecker::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'checkAgainstConstraintsOnEntityId', 'checkAgainstConstraintsOnClaimId' ] );
		$delegatingConstraintChecker = $mock->getMock();
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnEntityId' )
			->willReturn( [] );
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnClaimId' )
			->willReturn( [] );

		$result = $this->getResultsSource( $delegatingConstraintChecker )->getResults(
			[ new ItemId( self::NONEXISTENT_ITEM ) ],
			[ self::NONEXISTENT_CLAIM ],
			[],
			[ CheckResult::STATUS_TODO ]
		)->getArray();

		$this->assertSame( [], $result );
	}

	public function testGetResults_Empty_WithDefaultResults() {
		$mock = $this->getMockBuilder( DelegatingConstraintChecker::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'checkAgainstConstraintsOnEntityId', 'checkAgainstConstraintsOnClaimId' ] );
		$delegatingConstraintChecker = $mock->getMock();
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnEntityId' )
			->willReturnCallback( function (
				EntityId $entityId,
				array $constraintIds = null,
				callable $defaultResultsPerContext = null,
				callable $defaultResultsPerEntity = null
			) {
				if ( $defaultResultsPerEntity !== null ) {
					return $defaultResultsPerEntity( $entityId );
				} else {
					return [];
				}
			} );
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnClaimId' )
			->willReturn( [] );

		$result = $this->getResultsSource( $delegatingConstraintChecker )->getResults(
			[ new ItemId( self::NONEXISTENT_ITEM ) ],
			[ self::NONEXISTENT_CLAIM ],
			[],
			[ CheckResult::STATUS_TODO ]
		)->getArray();

		$this->assertCount( 1, $result );
		$this->assertSame( self::NONEXISTENT_ITEM, $result[0]->getContextCursor()->getEntityId() );
		$this->assertEquals(
			[ new ItemId( self::NONEXISTENT_ITEM ) ],
			$result[0]->getMetadata()->getDependencyMetadata()->getEntityIds()
		);
	}

	public function testGetResults_DependencyMetadata() {
		$mock = $this->getMockBuilder( DelegatingConstraintChecker::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'checkAgainstConstraintsOnEntityId', 'checkAgainstConstraintsOnClaimId' ] );
		$delegatingConstraintChecker = $mock->getMock();
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnEntityId' )
			->willReturn( [
				( new CheckResult(
					new MainSnakContext(
						new Item( new ItemId( 'Q1' ) ),
						new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
					),
					new Constraint(
						'P1$47681880-d5f5-417d-96c3-570d6e94d234',
						new NumericPropertyId( 'P1' ),
						'Q1',
						[]
					)
				) )->withMetadata( Metadata::ofDependencyMetadata(
					DependencyMetadata::ofEntityId( new ItemId( 'Q100' ) ) ) ),
			] );
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnClaimId' )
			->willReturn( [
				( new CheckResult(
					new MainSnakContext(
						new Item( new ItemId( 'Q2' ) ),
						new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
					),
					new Constraint(
						'P1$47681880-d5f5-417d-96c3-570d6e94d234',
						new NumericPropertyId( 'P1' ),
						'Q1',
						[]
					)
				) )->withMetadata( Metadata::ofDependencyMetadata(
					DependencyMetadata::ofEntityId( new NumericPropertyId( 'P100' ) ) ) ),
			] );

		$metadata = $this->getResultsSource( $delegatingConstraintChecker )->getResults(
			[ new ItemId( 'Q1' ) ],
			[ 'Q2$73408a9b-b1b0-4035-bf36-1e65ecf8772d' ],
			null,
			[ CheckResult::STATUS_TODO ]
		)->getMetadata();

		$expected = [ new ItemId( 'Q100' ), new NumericPropertyId( 'P100' ) ];
		$actual = $metadata->getDependencyMetadata()->getEntityIds();
		sort( $expected );
		sort( $actual );
		$this->assertEquals( $expected, $actual );
	}

	public function testGetResults_FilterStatuses() {
		$q1 = new ItemId( 'Q1' );
		$mock = $this->getMockBuilder( DelegatingConstraintChecker::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'checkAgainstConstraintsOnEntityId' ] );
		$delegatingConstraintChecker = $mock->getMock();
		$constraint = new Constraint(
			'P1$47681880-d5f5-417d-96c3-570d6e94d234',
			new NumericPropertyId( 'P1' ),
			'Q1',
			[]
		);
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnEntityId' )
			->willReturn( [
				new CheckResult(
					new MainSnakContext(
						new Item( $q1 ),
						new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
					),
					$constraint,
					CheckResult::STATUS_VIOLATION
				),
				new CheckResult(
					new MainSnakContext(
						new Item( $q1 ),
						new Statement( new PropertySomeValueSnak( new NumericPropertyId( 'P1' ) ) )
					),
					$constraint,
					CheckResult::STATUS_COMPLIANCE
				),
			] );

		$results = $this->getResultsSource( $delegatingConstraintChecker )->getResults(
			[ $q1 ],
			[],
			[],
			[ CheckResult::STATUS_VIOLATION ]
		)->getArray();

		$this->assertCount( 1, $results );
		$this->assertSame( CheckResult::STATUS_VIOLATION, $results[0]->getStatus() );
	}

	public function testGetResults_FilterStatuses_DependencyMetadata() {
		$mock = $this->getMockBuilder( DelegatingConstraintChecker::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'checkAgainstConstraintsOnEntityId' ] );
		$delegatingConstraintChecker = $mock->getMock();
		$context = new MainSnakContext(
			new Item( new ItemId( 'Q1' ) ),
			new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
		);
		$delegatingConstraintChecker->method( 'checkAgainstConstraintsOnEntityId' )
			->willReturn( [
				( new CheckResult(
					$context,
					new Constraint(
						'P1$47681880-d5f5-417d-96c3-570d6e94d234',
						new NumericPropertyId( 'P1' ),
						'Q1',
						[]
					),
					CheckResult::STATUS_COMPLIANCE
				) )->withMetadata( Metadata::ofDependencyMetadata(
					DependencyMetadata::ofEntityId( new ItemId( 'Q100' ) ) ) ),
				( new NullResult( $context->getCursor() ) ),
			] );

		$result = $this->getResultsSource( $delegatingConstraintChecker )->getResults(
			[ new ItemId( 'Q1' ) ],
			[],
			[],
			[ CheckResult::STATUS_VIOLATION ]
		);

		$this->assertCount(
			1,
			$result->getArray(),
			'real check results should have been filtered by status'
		);
		$this->assertInstanceOf(
			NullResult::class,
			$result->getArray()[0],
			'only check result left should be NullResult'
		);
		$this->assertEquals(
			Metadata::blank(),
			$result->getArray()[0]->getMetadata(),
			'NullResult should not have metadata'
		);
		$this->assertEquals(
			[ new ItemId( 'Q100' ) ],
			$result->getMetadata()->getDependencyMetadata()->getEntityIds(),
			'dependency metadata should still be there even though the check results were filtered'
		);
	}

}
