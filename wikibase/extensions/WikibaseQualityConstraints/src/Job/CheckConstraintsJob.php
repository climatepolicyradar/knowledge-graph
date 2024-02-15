<?php

namespace WikibaseQuality\ConstraintReport\Job;

use Job;
use MediaWiki\MediaWikiServices;
use MediaWiki\Title\Title;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\EntityIdParser;
use Wikibase\DataModel\Entity\EntityIdParsingException;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Api\CachingResultsSource;
use WikibaseQuality\ConstraintReport\ConstraintsServices;
use Wikimedia\Assert\Assert;

/**
 * A job that runs constraint checks for an item
 *
 * @author Jonas Kress
 * @license GPL-2.0-or-later
 */
class CheckConstraintsJob extends Job {

	public const COMMAND = 'constraintsRunCheck';

	/**
	 * @var CachingResultsSource
	 */
	private $resultsSource;

	/**
	 * @var EntityIdParser
	 */
	private $entityIdParser;

	/**
	 * @param Title $title
	 * @param string[] $params should contain 'entityId' => 'Q1234'
	 */
	public function __construct( Title $title, array $params ) {
		parent::__construct( self::COMMAND, $title, $params );
		$this->removeDuplicates = true;

		Assert::parameterType( 'string', $params['entityId'], '$params[\'entityId\']' );

		$resultSource = ConstraintsServices::getResultsSource( MediaWikiServices::getInstance() );
		'@phan-var CachingResultsSource $resultSource';
		// This job should only ever be used when caching result sources are used.
		$this->setResultsSource( $resultSource );

		$this->setEntityIdParser( WikibaseRepo::getEntityIdParser() );
	}

	public function setResultsSource( CachingResultsSource $resultsSource ) {
		$this->resultsSource = $resultsSource;
	}

	public function setEntityIdParser( EntityIdParser $parser ) {
		$this->entityIdParser = $parser;
	}

	/**
	 * @see Job::run
	 *
	 * @return bool
	 */
	public function run() {
		try {
			$entityId = $this->entityIdParser->parse( $this->params['entityId'] );
		} catch ( EntityIdParsingException $e ) {
			return false;
		}

		$this->checkConstraints( $entityId );

		return true;
	}

	private function checkConstraints( EntityId $entityId ) {
		$this->resultsSource->getResults(
			[ $entityId ],
			[],
			null,
			[]
		);
	}

}
