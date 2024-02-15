<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Cache;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\Tests\Fake\AppendingContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachedCheckResultsTest extends \MediaWikiUnitTestCase {

	public function testFiltered() {
		$p1 = new NumericPropertyId( 'P1' );
		$q1 = new ItemId( 'Q1' );
		$constraint = new Constraint( 'constraint ID', $p1, 'constraint type', [] );
		$result1 = ( new CheckResult(
			new AppendingContextCursor(),
			$constraint,
			CheckResult::STATUS_COMPLIANCE
		) )->withMetadata(
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $p1 ) )
		);
		$result2 = ( new CheckResult(
			new AppendingContextCursor(),
			$constraint,
			CheckResult::STATUS_VIOLATION,
			new ViolationMessage( 'wbqc-violation-message-single-value' )
		) )->withMetadata(
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q1 ) )
		);

		$cachedCheckResults = new CachedCheckResults(
			[ $result2 ],
			Metadata::merge( [ $result1->getMetadata(), $result2->getMetadata() ] )
		);
		$checkResults = $cachedCheckResults->getArray();
		$metadata = $cachedCheckResults->getMetadata();

		$this->assertSame( [ $result2 ], $checkResults );
		$this->assertSame( [ $p1, $q1 ], $metadata->getDependencyMetadata()->getEntityIds() );
	}

}
