<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit;

use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintSerializer;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintSerializer
 * @covers WikibaseQuality\ConstraintReport\ConstraintDeserializer
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ConstraintSerializationTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider provideConstraints
	 */
	public function testSerialize( Constraint $constraint, array $serialization ) {
		$serializer = new ConstraintSerializer();

		$serialized = $serializer->serialize( $constraint );

		$this->assertSame( $serialization, $serialized );
	}

	/**
	 * @dataProvider provideConstraints
	 */
	public function testSerialize_withoutConstraintParameters( Constraint $constraint, array $serialization ) {
		$serializer = new ConstraintSerializer( false );

		$serialized = $serializer->serialize( $constraint );

		unset( $serialization['params'] );
		$this->assertSame( $serialization, $serialized );
	}

	/**
	 * @dataProvider provideConstraints
	 */
	public function testDeserialize( Constraint $constraint, array $serialization ) {
		$deserializer = new ConstraintDeserializer();

		$deserialized = $deserializer->deserialize( $serialization );

		$this->assertEquals( $constraint, $deserialized );
	}

	/**
	 * @dataProvider provideConstraints
	 */
	public function testDeserialize_withoutConstraintParameters( Constraint $constraint, array $serialization ) {
		$deserializer = new ConstraintDeserializer();
		unset( $serialization['params'] );

		$deserialized = $deserializer->deserialize( $serialization );

		$expected = new Constraint(
			$constraint->getConstraintId(),
			$constraint->getPropertyId(),
			$constraint->getConstraintTypeItemId(),
			[]
		);
		$this->assertEquals( $expected, $deserialized );
	}

	public static function provideConstraints() {
		$constraintId = 'P569$EF034B4A-6C21-4199-A6FE-6F36B28FCDAE';
		$propertyIdSerialization = 'P569';
		$constraintTypeItemId = 'Q21502838';
		$constraintParameters = [ 'P2306' => [ [
			'snaktype' => 'value',
			'property' => 'P2306',
			'hash' => '632db79c2fd121822c892d35903b82fdb9e82d1d',
			'datavalue' => [
				'value' => [ 'entity-type' => 'property', 'numeric-id' => 625, 'id' => 'P625 ' ],
				'type' => 'wikibase-entityid',
			],
		] ] ];
		yield 'date of birth conflicts with coordinate location' => [
			new Constraint(
				$constraintId,
				new NumericPropertyId( $propertyIdSerialization ),
				$constraintTypeItemId,
				$constraintParameters
			),
			[
				'id' => $constraintId,
				'pid' => $propertyIdSerialization,
				'qid' => $constraintTypeItemId,
				'params' => $constraintParameters,
			],
		];
	}

}
