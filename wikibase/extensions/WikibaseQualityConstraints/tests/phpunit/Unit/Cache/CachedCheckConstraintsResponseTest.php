<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Cache;

use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckConstraintsResponse;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckConstraintsResponse
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachedCheckConstraintsResponseTest extends \MediaWikiUnitTestCase {

	public function testGetArray() {
		$array = [ 'boolean' => true ];
		$cm = Metadata::blank();

		$cqr = new CachedCheckConstraintsResponse( $array, $cm );

		$this->assertSame( $array, $cqr->getArray() );
	}

}
