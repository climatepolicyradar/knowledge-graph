<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Checker\ValueCountChecker;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ValueCountCheckerHelper;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ValueCountCheckerHelper
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ValueCountCheckerHelperTest extends \MediaWikiUnitTestCase {

	/**
	 * @dataProvider getPropertyCountProvider
	 */
	public function testGetPropertyCount( $snaks, $propertyIdSerialization, $expectedCount ) {
		$propertyId = new NumericPropertyId( $propertyIdSerialization );
		$helper = new ValueCountCheckerHelper();
		$propertyCount = $helper->getPropertyCount( $snaks, $propertyId );

		$this->assertSame( $expectedCount, $propertyCount );
	}

	public static function getPropertyCountProvider() {
		$p1_1 = new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) );
		$p1_2 = new PropertySomeValueSnak( new NumericPropertyId( 'P1' ) );
		$p2 = new PropertyNoValueSnak( new NumericPropertyId( 'P2' ) );

		return [
			'empty' => [ [], 'P1', 0 ],
			'only other property' => [ [ $p2 ], 'P1', 0 ],
			'only searched property' => [ [ $p1_1 ], 'P1', 1 ],
			'both properties' => [ [ $p1_1, $p2 ], 'P1', 1 ],
			'searched property multiple times' => [ [ $p1_1, $p1_2 ], 'P1', 2 ],
			'same snak multiple times' => [ [ $p1_1, $p1_1, $p1_2 ], 'P1', 3 ],
		];
	}

}
