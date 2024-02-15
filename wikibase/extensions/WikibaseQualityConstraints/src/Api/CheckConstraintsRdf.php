<?php

namespace WikibaseQuality\ConstraintReport\Api;

use Article;
use FormlessAction;
use IContextSource;
use Wikibase\Lib\Store\EntityIdLookup;
use Wikibase\Repo\Rdf\RdfVocabulary;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;
use WikibaseQuality\ConstraintReport\ConstraintsServices;
use Wikimedia\Purtle\RdfWriterFactory;

/**
 * Produce constraint check results in RDF.
 * Only returns cached constraint check results for now.
 *
 * @license GPL-2.0-or-later
 */
class CheckConstraintsRdf extends FormlessAction {

	/**
	 * @var EntityIdLookup
	 */
	private $entityIdLookup;
	/**
	 * @var ResultsSource
	 */
	private $resultsSource;
	/**
	 * @var RdfVocabulary
	 */
	private $rdfVocabulary;

	/**
	 * @param Article $page
	 * @param IContextSource $context
	 * @param ResultsSource $resultsSource
	 * @param EntityIdLookup $entityIdLookup
	 * @param RdfVocabulary $rdfVocabulary
	 */
	public function __construct(
		object $page,
		IContextSource $context,
		ResultsSource $resultsSource,
		EntityIdLookup $entityIdLookup,
		RdfVocabulary $rdfVocabulary
	) {
		parent::__construct( $page, $context );
		$this->resultsSource = $resultsSource;
		$this->entityIdLookup = $entityIdLookup;
		$this->rdfVocabulary = $rdfVocabulary;
	}

	/**
	 * @param Article $page
	 * @param IContextSource $context
	 * @return CheckConstraintsRdf
	 */
	public static function newFromGlobalState(
		object $page,
		IContextSource $context
	) {
		return new static(
			$page,
			$context,
			ConstraintsServices::getResultsSource(),
			WikibaseRepo::getEntityIdLookup(),
			WikibaseRepo::getRdfVocabulary()
		);
	}

	/**
	 * Return the name of the action this object responds to
	 * @since 1.17
	 *
	 * @return string Lowercase name
	 */
	public function getName() {
		return 'constraintsrdf';
	}

	/**
	 * Whether this action requires the wiki not to be locked
	 * @since 1.17
	 *
	 * @return bool
	 */
	public function requiresWrite() {
		return false;
	}

	/**
	 * @see Action::requiresUnblock
	 *
	 * @return bool Always false.
	 */
	public function requiresUnblock() {
		return false;
	}

	/**
	 * Cleanup GUID string so it's OK for RDF.
	 * Should match what we're doing on RDF generation.
	 * @param string $guid
	 * @return string
	 */
	private function cleanupGuid( $guid ) {
		return preg_replace( '/[^\w-]/', '-', $guid );
	}

	/**
	 * Show something on GET request.
	 * @return string|null Will be added to the HTMLForm if present, or just added to the
	 *     output if not.  Return null to not add anything
	 */
	public function onView() {
		$response = $this->getRequest()->response();
		$this->getOutput()->disable();

		if ( !$this->resultsSource instanceof CachingResultsSource ) {
			// TODO: make configurable whether only cached results are returned
			$response->statusHeader( 501 ); // Not Implemented
			return null;
		}

		$entityId = $this->entityIdLookup->getEntityIdForTitle( $this->getTitle() );
		if ( $entityId === null ) {
			$response->statusHeader( 404 ); // Not Found
			return null;
		}
		$revId = $this->getRequest()->getInt( 'revision' );

		$results = $this->resultsSource->getStoredResults( $entityId, $revId );
		if ( $results === null ) {
			$response->statusHeader( 204 ); // No Content
			return null;
		}

		$format = 'ttl'; // TODO: make format an option

		$writerFactory = new RdfWriterFactory();
		$formatName = $writerFactory->getFormatName( $format );
		$contentType = $writerFactory->getMimeTypes( $formatName )[0];

		$writer = $writerFactory->getWriter( $formatName );
		foreach ( [ RdfVocabulary::NS_STATEMENT, RdfVocabulary::NS_ONTOLOGY ] as $ns ) {
			$writer->prefix( $ns, $this->rdfVocabulary->getNamespaceURI( $ns ) );
		}
		$writer->start();
		$writtenAny = false;

		foreach ( $results->getArray() as $checkResult ) {
			if ( $checkResult instanceof NullResult ) {
				continue;
			}
			if ( $checkResult->getStatus() === CheckResult::STATUS_BAD_PARAMETERS ) {
				continue;
			}
			$writtenAny = true;
			$writer->about( RdfVocabulary::NS_STATEMENT,
				$this->cleanupGuid( $checkResult->getContextCursor()->getStatementGuid() ) )
				->say( RdfVocabulary::NS_ONTOLOGY, 'hasViolationForConstraint' )
				->is( RdfVocabulary::NS_STATEMENT,
					$this->cleanupGuid( $checkResult->getConstraint()->getConstraintId() ) );
		}
		$writer->finish();
		if ( $writtenAny ) {
			$response->header( "Content-Type: $contentType; charset=UTF-8" );
			echo $writer->drain();
		} else {
			// Do not output RDF if we haven't written any actual statements. Output 204 instead
			$writer->drain();
			$response->statusHeader( 204 ); // No Content
		}
		return null;
	}

}
