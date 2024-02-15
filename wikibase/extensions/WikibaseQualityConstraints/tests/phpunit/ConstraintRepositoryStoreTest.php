<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\Tests\Store\Sql\Terms\Util\FakeLoadBalancer;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintRepositoryStore;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintRepositoryStore
 *
 * @group WikibaseQualityConstraints
 * @group Database
 * @group medium
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ConstraintRepositoryStoreTest extends \MediaWikiIntegrationTestCase {

	private function newConstraintRepositoryStore() {
		return new ConstraintRepositoryStore(
			new FakeLoadBalancer( [ 'dbr' => $this->db ] ),
			false
		);
	}

	public function testInsertBatch() {
		$this->insertTestData();

		$constraints = [
			new Constraint( 'foo', new NumericPropertyId( 'P42' ), 'TestConstraint', [ 'foo' => 'bar' ] ),
			new Constraint( 'bar', new NumericPropertyId( 'P42' ), 'TestConstraint', [ 'bar' => 'baz' ] ),
			new Constraint( 'baz', new NumericPropertyId( 'P42' ), 'TestConstraint', [] ),
		];
		$repo = $this->newConstraintRepositoryStore();
		$repo->insertBatch( $constraints );

		$this->assertSelect(
			'wbqc_constraints',
			[
				'constraint_guid',
				'pid',
				'constraint_type_qid',
				'constraint_parameters',
			],
			[],
			[
				[
					'1',
					1,
					'Multi value',
					'{}',
				],
				[
					'3',
					1,
					'Single value',
					'{}',
				],
				[
					'bar',
					'42',
					'TestConstraint',
					'{"bar":"baz"}',
				],
				[
					'baz',
					'42',
					'TestConstraint',
					'{}',
				],
				[
					'foo',
					'42',
					'TestConstraint',
					'{"foo":"bar"}',
				],
			]
		);
	}

	public function testInsertBatchTooLongParameters() {
		$this->db->delete( 'wbqc_constraints', '*' );

		$constraintParameters = [ 'known_exception' => [] ];
		for ( $i = 0; $i < 10000; $i++ ) {
			$constraintParameters['known_exception'][] = [
				'snaktype' => 'value', 'property' => 'P2303',
				'hash' => '1deb3a3ba50cbbf2672b0def6c9e96bcf3f533e5', 'datavalue' => [
					'value' => [
						'entity-type' => 'item', 'numeric-id' => 2961108, 'id' => 'Q2961108',
					],
					'type' => 'wikibase-entityid',
				],
			];
		}

		$repo = $this->newConstraintRepositoryStore();
		$repo->insertBatch( [
			new Constraint(
				'P1$13510cdc-0f91-4ea3-b71d-db2a33c27dff',
				new NumericPropertyId( 'P1' ),
				'Q1',
				$constraintParameters
			),
		] );

		$this->assertSelect(
			'wbqc_constraints',
			[
				'constraint_guid',
				'pid',
				'constraint_type_qid',
				'constraint_parameters',
			],
			[],
			[
				[
					'P1$13510cdc-0f91-4ea3-b71d-db2a33c27dff',
					'1',
					'Q1',
					'{"@error":{"toolong":true}}',
				],
			]
		);
	}

	private function insertTestData() {
		$this->db->delete( 'wbqc_constraints', '*' );
		$this->db->insert( 'wbqc_constraints', [
			[
				'constraint_guid' => '1',
				'pid' => 1,
				'constraint_type_qid' => 'Multi value',
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => '3',
				'pid' => 1,
				'constraint_type_qid' => 'Single value',
				'constraint_parameters' => '{}',
			],
		] );
	}

}
