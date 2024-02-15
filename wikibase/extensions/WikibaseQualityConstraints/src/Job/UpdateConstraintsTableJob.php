<?php

namespace WikibaseQuality\ConstraintReport\Job;

use Config;
use Job;
use JobQueueGroup;
use MediaWiki\MediaWikiServices;
use MediaWiki\Title\Title;
use Serializers\Serializer;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Entity\Property;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\Lib\Store\EntityRevisionLookup;
use Wikibase\Lib\Store\LookupConstants;
use Wikibase\Repo\Store\Store;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintsServices;
use WikibaseQuality\ConstraintReport\ConstraintStore;
use Wikimedia\Assert\Assert;
use Wikimedia\Rdbms\ILBFactory;

/**
 * A job that updates the constraints table
 * when changes were made on a property.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class UpdateConstraintsTableJob extends Job {

	/**
	 * How many constraints to write in one transaction before waiting for replication.
	 * Properties with more constraints than this will not be updated atomically
	 * (they will appear to have an incomplete set of constraints for a time).
	 */
	private const BATCH_SIZE = 50;

	public static function newFromGlobalState( Title $title, array $params ) {
		Assert::parameterType( 'string', $params['propertyId'], '$params["propertyId"]' );
		$services = MediaWikiServices::getInstance();
		return new UpdateConstraintsTableJob(
			$title,
			$params,
			$params['propertyId'],
			$params['revisionId'] ?? null,
			$services->getMainConfig(),
			ConstraintsServices::getConstraintStore(),
			$services->getDBLoadBalancerFactory(),
			WikibaseRepo::getStore()->getEntityRevisionLookup( Store::LOOKUP_CACHING_DISABLED ),
			WikibaseRepo::getBaseDataModelSerializerFactory( $services )
				->newSnakSerializer(),
			$services->getJobQueueGroup()
		);
	}

	/**
	 * @var string
	 */
	private $propertyId;

	/**
	 * @var int|null
	 */
	private $revisionId;

	/**
	 * @var Config
	 */
	private $config;

	/**
	 * @var ConstraintStore
	 */
	private $constraintStore;

	/** @var ILBFactory */
	private $lbFactory;

	/**
	 * @var EntityRevisionLookup
	 */
	private $entityRevisionLookup;

	/**
	 * @var Serializer
	 */
	private $snakSerializer;

	/**
	 * @var JobQueueGroup
	 */
	private $jobQueueGroup;

	/**
	 * @param Title $title
	 * @param string[] $params should contain 'propertyId' => 'P...'
	 * @param string $propertyId property ID of the property for this job (which has the constraint statements)
	 * @param int|null $revisionId revision ID that triggered this job, if any
	 * @param Config $config
	 * @param ConstraintStore $constraintStore
	 * @param ILBFactory $lbFactory
	 * @param EntityRevisionLookup $entityRevisionLookup
	 * @param Serializer $snakSerializer
	 * @param JobQueueGroup $jobQueueGroup
	 */
	public function __construct(
		Title $title,
		array $params,
		$propertyId,
		$revisionId,
		Config $config,
		ConstraintStore $constraintStore,
		ILBFactory $lbFactory,
		EntityRevisionLookup $entityRevisionLookup,
		Serializer $snakSerializer,
		JobQueueGroup $jobQueueGroup
	) {
		parent::__construct( 'constraintsTableUpdate', $title, $params );

		$this->propertyId = $propertyId;
		$this->revisionId = $revisionId;
		$this->config = $config;
		$this->constraintStore = $constraintStore;
		$this->lbFactory = $lbFactory;
		$this->entityRevisionLookup = $entityRevisionLookup;
		$this->snakSerializer = $snakSerializer;
		$this->jobQueueGroup = $jobQueueGroup;
	}

	public function extractParametersFromQualifiers( SnakList $qualifiers ) {
		$parameters = [];
		foreach ( $qualifiers as $qualifier ) {
			$qualifierId = $qualifier->getPropertyId()->getSerialization();
			$paramSerialization = $this->snakSerializer->serialize( $qualifier );
			$parameters[$qualifierId][] = $paramSerialization;
		}
		return $parameters;
	}

	public function extractConstraintFromStatement(
		NumericPropertyId $propertyId,
		Statement $constraintStatement
	) {
		$constraintId = $constraintStatement->getGuid();
		'@phan-var string $constraintId'; // we know the statement has a non-null GUID
		$snak = $constraintStatement->getMainSnak();
		'@phan-var \Wikibase\DataModel\Snak\PropertyValueSnak $snak';
		$dataValue = $snak->getDataValue();
		'@phan-var \Wikibase\DataModel\Entity\EntityIdValue $dataValue';
		$entityId = $dataValue->getEntityId();
		$constraintTypeQid = $entityId->getSerialization();
		$parameters = $this->extractParametersFromQualifiers( $constraintStatement->getQualifiers() );
		return new Constraint(
			$constraintId,
			$propertyId,
			$constraintTypeQid,
			$parameters
		);
	}

	public function importConstraintsForProperty(
		Property $property,
		ConstraintStore $constraintStore,
		NumericPropertyId $propertyConstraintPropertyId
	) {
		$constraintsStatements = $property->getStatements()
			->getByPropertyId( $propertyConstraintPropertyId )
			->getByRank( [ Statement::RANK_PREFERRED, Statement::RANK_NORMAL ] );
		$constraints = [];
		foreach ( $constraintsStatements->getIterator() as $constraintStatement ) {
			// @phan-suppress-next-line PhanTypeMismatchArgumentSuperType
			$constraints[] = $this->extractConstraintFromStatement( $property->getId(), $constraintStatement );
			if ( count( $constraints ) >= self::BATCH_SIZE ) {
				$constraintStore->insertBatch( $constraints );
				// interrupt transaction and wait for replication
				$connection = $this->lbFactory->getMainLB()->getConnection( DB_PRIMARY );
				$connection->endAtomic( __CLASS__ );
				if ( !$connection->explicitTrxActive() ) {
					$this->lbFactory->waitForReplication();
				}
				$connection->startAtomic( __CLASS__ );
				$constraints = [];
			}
		}
		$constraintStore->insertBatch( $constraints );
	}

	/**
	 * @see Job::run
	 *
	 * @return bool
	 */
	public function run() {
		// TODO in the future: only touch constraints affected by the edit (requires T163465)

		$propertyId = new NumericPropertyId( $this->propertyId );
		$propertyRevision = $this->entityRevisionLookup->getEntityRevision(
			$propertyId,
			0, // latest
			LookupConstants::LATEST_FROM_REPLICA
		);

		if ( $this->revisionId !== null && $propertyRevision->getRevisionId() < $this->revisionId ) {
			$this->jobQueueGroup->push( $this );
			return true;
		}

		$connection = $this->lbFactory->getMainLB()->getConnection( DB_PRIMARY );
		// start transaction (if not started yet) – using __CLASS__, not __METHOD__,
		// because importConstraintsForProperty() can interrupt the transaction
		$connection->startAtomic( __CLASS__ );

		$this->constraintStore->deleteForProperty( $propertyId );

		/** @var Property $property */
		$property = $propertyRevision->getEntity();
		'@phan-var Property $property';
		$this->importConstraintsForProperty(
			$property,
			$this->constraintStore,
			new NumericPropertyId( $this->config->get( 'WBQualityConstraintsPropertyConstraintId' ) )
		);

		$connection->endAtomic( __CLASS__ );

		return true;
	}

}
