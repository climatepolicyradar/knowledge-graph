<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Entity\Property;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Tests\NewStatement;
use Wikibase\Lib\Changes\Change;
use Wikibase\Lib\Changes\EntityChange;
use Wikibase\Lib\Tests\Changes\TestChanges;
use WikibaseQuality\ConstraintReport\WikibaseQualityConstraintsHooks;

/**
 * @covers WikibaseQuality\ConstraintReport\WikibaseQualityConstraintsHooks
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class WikibaseQualityConstraintsHooksTest extends \PHPUnit\Framework\TestCase {

	use DefaultConfig;

	/**
	 * @dataProvider provideChanges
	 * @param Change $change
	 * @param bool $expected
	 */
	public function testIsConstraintStatementsChange( Change $change, $expected ) {
		$actual = WikibaseQualityConstraintsHooks::isConstraintStatementsChange(
			self::getDefaultConfig(),
			$change
		);
		$this->assertSame( $expected, $actual );
	}

	public static function provideChanges(): iterable {
		$factory = TestChanges::getEntityChangeFactory();
		$changes = TestChanges::getChanges();
		$changeKeys = [];

		// changes on items
		$changeKeys += [
			'item-creation',
			'set-dewiki-sitelink',
			'change-sitelink-order',
			'set-en-aliases',
			'add-claim',
			'remove-claim',
		];

		// changes on properties not affecting the statements
		$changeKeys += [
			'property-creation',
			'property-set-label',
		];

		foreach ( $changeKeys as $changeKey ) {
			yield $changeKey => [ $changes[$changeKey], false ];
		}

		// (TestChanges doesn’t have pre-defined changes for the following)
		$new = new Property( new NumericPropertyId( 'P1' ), null, 'string' );
		$old = $new->copy();

		// changes on properties affecting non-constraint statements
		$new->getStatements()->addStatement( NewStatement::noValueFor( 'P1' )->build() );
		$change = $factory->newFromUpdate( EntityChange::UPDATE, $old, $new );
		yield 'property-add-claim' => [ $change, false ];
		$old = $new->copy();

		$new->getStatements()->clear();
		$change = $factory->newFromUpdate( EntityChange::UPDATE, $old, $new );
		yield 'property-remove-claim' => [ $change, false ];
		$old = $new->copy();

		// changes on properties affecting constraint statements
		$p2302 = self::getDefaultConfig()->get( 'WBQualityConstraintsPropertyConstraintId' );
		$statement = NewStatement::noValueFor( $p2302 )->build();
		$new->getStatements()->addStatement( $statement );
		$change = $factory->newFromUpdate( EntityChange::UPDATE, $old, $new );
		yield 'property-add-constraint' => [ $change, true ];
		$old = $new->copy();

		$parameter = NewStatement::noValueFor( 'P1' )->build()->getMainSnak();
		$statement->setQualifiers( new SnakList( [ $parameter ] ) );
		$change = $factory->newFromUpdate( EntityChange::UPDATE, $old, $new );
		yield 'property-add-constraint-parameter' => [ $change, true ];
		$old = $new->copy();

		$statement->setQualifiers( new SnakList() );
		$change = $factory->newFromUpdate( EntityChange::UPDATE, $old, $new );
		yield 'property-remove-constraint-parameter' => [ $change, true ];
		$old = $new->copy();

		$new->getStatements()->addStatement(
			NewStatement::noValueFor( $p2302 )
				->withSomeGuid()
				->build()
		);
		$change = $factory->newFromUpdate( EntityChange::UPDATE, $old, $new );
		yield 'property-copy-constraint' => [ $change, true ];
	}

}
