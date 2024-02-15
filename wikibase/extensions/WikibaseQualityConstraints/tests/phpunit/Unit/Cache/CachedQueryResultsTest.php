<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Cache;

use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedQueryResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedQueryResults
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachedQueryResultsTest extends \MediaWikiUnitTestCase {

	public function testGetArray() {
		$array = [ 'boolean' => true ];
		$cm = Metadata::blank();

		$cqr = new CachedQueryResults( $array, $cm );

		$this->assertSame( $array, $cqr->getArray() );
	}

}
