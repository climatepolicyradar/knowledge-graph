<?php

namespace WikibaseQuality\ConstraintReport\Tests\Job;

use Job;
use MediaWiki\Title\Title;
use MediaWikiIntegrationTestCase;
use Wikibase\DataModel\Entity\ItemId;
use WikibaseQuality\ConstraintReport\Api\CachingResultsSource;
use WikibaseQuality\ConstraintReport\Job\CheckConstraintsJob;

/**
 * @covers \WikibaseQuality\ConstraintReport\Job\CheckConstraintsJob
 *
 * @group WikibaseQualityConstraints
 * @group Database
 * @group medium
 *
 * @author Jonas Kress
 * @license GPL-2.0-or-later
 */
class CheckConstraintsJobTest extends MediaWikiIntegrationTestCase {

	private const ENTITY_ID = 'Q101';
	private const JOB_TITLE_STRING = 'CheckConstraintsJobTitleString';

	private function createJob( $titleString, $params ) {
		$title = Title::makeTitle( NS_MAIN, $titleString );
		return new CheckConstraintsJob( $title, $params );
	}

	public function testCreationFromFactory() {
		Job::factory(
			CheckConstraintsJob::COMMAND,
			Title::newMainPage(),
			[ 'entityId' => 'Q123' ]
		);
		$this->assertTrue( true ); // No exception
	}

	public function testJobDuplicationInfo() {
		$params = [
			'entityId' => self::ENTITY_ID,
			'namespace' => NS_MAIN,
			'title' => self::JOB_TITLE_STRING,
		];
		$job = $this->createJob( self::JOB_TITLE_STRING, $params );

		$this->assertSame(
			[
				'type' => CheckConstraintsJob::COMMAND,
				'params' => $params,
			],
			$job->getDeduplicationInfo()
		);

		$this->assertTrue( $job->ignoreDuplicates() );
	}

	public function testJobRun() {
		$resultSource = $this->createMock( CachingResultsSource::class );
		$resultSource->expects( $this->once() )
			->method( 'getResults' )
			->with( [ new ItemId( self::ENTITY_ID ) ] );

		$job = $this->createJob( self::JOB_TITLE_STRING, [ 'entityId' => self::ENTITY_ID ] );
		$job->setResultsSource( $resultSource );

		$job->run();
	}

}
