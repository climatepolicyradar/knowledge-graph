<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Config;
use DataValues\DataValue;
use DataValues\MonolingualTextValue;
use DateInterval;
use FormatJson;
use IBufferingStatsdDataFactory;
use InvalidArgumentException;
use MapCacheLRU;
use MediaWiki\Http\HttpRequestFactory;
use MWHttpRequest;
use UnexpectedValueException;
use WANObjectCache;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\EntityIdParser;
use Wikibase\DataModel\Entity\EntityIdParsingException;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\PropertyId;
use Wikibase\DataModel\Services\Lookup\PropertyDataTypeLookup;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\Repo\Rdf\RdfVocabulary;
use WikibaseQuality\ConstraintReport\Api\ExpiryLock;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedBool;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedEntityIds;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedQueryResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\Role;
use Wikimedia\Timestamp\ConvertibleTimestamp;

/**
 * Class for running a SPARQL query on some endpoint and getting the results.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class SparqlHelper {

	/**
	 * @var RdfVocabulary
	 */
	private $rdfVocabulary;

	/**
	 * @var string[]
	 */
	private $entityPrefixes;

	/**
	 * @var string
	 */
	private $prefixes;

	/**
	 * @var EntityIdParser
	 */
	private $entityIdParser;

	/**
	 * @var PropertyDataTypeLookup
	 */
	private $propertyDataTypeLookup;

	/**
	 * @var WANObjectCache
	 */
	private $cache;

	/**
	 * @var ViolationMessageSerializer
	 */
	private $violationMessageSerializer;

	/**
	 * @var ViolationMessageDeserializer
	 */
	private $violationMessageDeserializer;

	/**
	 * @var IBufferingStatsdDataFactory
	 */
	private $dataFactory;

	/**
	 * @var LoggingHelper
	 */
	private $loggingHelper;

	/**
	 * @var string
	 */
	private $defaultUserAgent;

	/**
	 * @var ExpiryLock
	 */
	private $throttlingLock;

	/**
	 * @var int stands for: No Retry-After header-field was sent back
	 */
	private const NO_RETRY_AFTER = -1;
	/**
	 * @var int stands for: Empty Retry-After header-field was sent back
	 */
	private const EMPTY_RETRY_AFTER = -2;
	/**
	 * @var int stands for: Invalid Retry-After header-field was sent back
	 * link a string
	 */
	private const INVALID_RETRY_AFTER = -3;
	/**
	 * @var string ID on which the lock is applied on
	 */
	public const EXPIRY_LOCK_ID = 'SparqlHelper.runQuery';

	/**
	 * @var int HTTP response code for too many requests
	 */
	private const HTTP_TOO_MANY_REQUESTS = 429;

	/**
	 * @var HttpRequestFactory
	 */
	private $requestFactory;

	// config variables

	/**
	 * @var string
	 */
	private $endpoint;

	/**
	 * @var int
	 */
	private $maxQueryTimeMillis;

	/**
	 * @var string
	 */
	private $instanceOfId;

	/**
	 * @var string
	 */
	private $subclassOfId;

	/**
	 * @var int
	 */
	private $cacheMapSize;

	/**
	 * @var string[]
	 */
	private $timeoutExceptionClasses;

	/**
	 * @var bool
	 */
	private $sparqlHasWikibaseSupport;

	/**
	 * @var int
	 */
	private $sparqlThrottlingFallbackDuration;

	public function __construct(
		Config $config,
		RdfVocabulary $rdfVocabulary,
		EntityIdParser $entityIdParser,
		PropertyDataTypeLookup $propertyDataTypeLookup,
		WANObjectCache $cache,
		ViolationMessageSerializer $violationMessageSerializer,
		ViolationMessageDeserializer $violationMessageDeserializer,
		IBufferingStatsdDataFactory $dataFactory,
		ExpiryLock $throttlingLock,
		LoggingHelper $loggingHelper,
		$defaultUserAgent,
		HttpRequestFactory $requestFactory
	) {
		$this->rdfVocabulary = $rdfVocabulary;
		$this->entityIdParser = $entityIdParser;
		$this->propertyDataTypeLookup = $propertyDataTypeLookup;
		$this->cache = $cache;
		$this->violationMessageSerializer = $violationMessageSerializer;
		$this->violationMessageDeserializer = $violationMessageDeserializer;
		$this->dataFactory = $dataFactory;
		$this->throttlingLock = $throttlingLock;
		$this->loggingHelper = $loggingHelper;
		$this->defaultUserAgent = $defaultUserAgent;
		$this->requestFactory = $requestFactory;
		$this->entityPrefixes = [];
		foreach ( $rdfVocabulary->entityNamespaceNames as $namespaceName ) {
			$this->entityPrefixes[] = $rdfVocabulary->getNamespaceURI( $namespaceName );
		}

		$this->endpoint = $config->get( 'WBQualityConstraintsSparqlEndpoint' );
		$this->maxQueryTimeMillis = $config->get( 'WBQualityConstraintsSparqlMaxMillis' );
		$this->instanceOfId = $config->get( 'WBQualityConstraintsInstanceOfId' );
		$this->subclassOfId = $config->get( 'WBQualityConstraintsSubclassOfId' );
		$this->cacheMapSize = $config->get( 'WBQualityConstraintsFormatCacheMapSize' );
		$this->timeoutExceptionClasses = $config->get(
			'WBQualityConstraintsSparqlTimeoutExceptionClasses'
		);
		$this->sparqlHasWikibaseSupport = $config->get(
			'WBQualityConstraintsSparqlHasWikibaseSupport'
		);
		$this->sparqlThrottlingFallbackDuration = (int)$config->get(
			'WBQualityConstraintsSparqlThrottlingFallbackDuration'
		);

		$this->prefixes = $this->getQueryPrefixes( $rdfVocabulary );
	}

	private function getQueryPrefixes( RdfVocabulary $rdfVocabulary ) {
		// TODO: it would probably be smarter that RdfVocubulary exposed these prefixes somehow
		$prefixes = '';
		foreach ( $rdfVocabulary->entityNamespaceNames as $sourceName => $namespaceName ) {
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
		}
		$prefixes .= <<<END
PREFIX wds: <{$rdfVocabulary->getNamespaceURI( RdfVocabulary::NS_STATEMENT )}>
PREFIX wdv: <{$rdfVocabulary->getNamespaceURI( RdfVocabulary::NS_VALUE )}>\n
END;

		foreach ( $rdfVocabulary->propertyNamespaceNames as $sourceName => $sourceNamespaces ) {
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_DIRECT_CLAIM];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_CLAIM];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_CLAIM_STATEMENT];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_QUALIFIER];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_QUALIFIER_VALUE];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_REFERENCE];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
			$namespaceName = $sourceNamespaces[RdfVocabulary::NSP_REFERENCE_VALUE];
			$prefixes .= <<<END
PREFIX {$namespaceName}: <{$rdfVocabulary->getNamespaceURI( $namespaceName )}>\n
END;
		}
		$prefixes .= <<<END
PREFIX wikibase: <{$rdfVocabulary->getNamespaceURI( RdfVocabulary::NS_ONTOLOGY )}>\n
END;
		return $prefixes;
	}

	/**
	 * @param string $id entity ID serialization of the entity to check
	 * @param string[] $classes entity ID serializations of the expected types
	 *
	 * @return CachedBool
	 * @throws SparqlHelperException if the query times out or some other error occurs
	 */
	public function hasType( $id, array $classes ) {
		// TODO hint:gearing is a workaround for T168973 and can hopefully be removed eventually
		$gearingHint = $this->sparqlHasWikibaseSupport ?
			' hint:Prior hint:gearing "forward".' :
			'';

		$metadatas = [];

		foreach ( array_chunk( $classes, 20 ) as $classesChunk ) {
			$classesValues = implode( ' ', array_map(
				static function ( $class ) {
					return 'wd:' . $class;
				},
				$classesChunk
			) );

			$query = <<<EOF
ASK {
  BIND(wd:$id AS ?item)
  VALUES ?class { $classesValues }
  ?item wdt:{$this->subclassOfId}* ?class.$gearingHint
}
EOF;

			$result = $this->runQuery( $query );
			$metadatas[] = $result->getMetadata();
			if ( $result->getArray()['boolean'] ) {
				return new CachedBool(
					true,
					Metadata::merge( $metadatas )
				);
			}
		}

		return new CachedBool(
			false,
			Metadata::merge( $metadatas )
		);
	}

	/**
	 * Helper function used by findEntitiesWithSameStatement to filter
	 * out entities with different qualifiers or no qualifier value.
	 *
	 * @param PropertyId $separator
	 * @return string
	 */
	private function nestedSeparatorFilter( PropertyId $separator ) {
		$filter = <<<EOF
  MINUS {
    ?statement pq:$separator ?qualifier.
    FILTER NOT EXISTS {
      ?otherStatement pq:$separator ?qualifier.
    }
  }
  MINUS {
    ?otherStatement pq:$separator ?qualifier.
    FILTER NOT EXISTS {
      ?statement pq:$separator ?qualifier.
    }
  }
  MINUS {
    ?statement a wdno:$separator.
    FILTER NOT EXISTS {
      ?otherStatement a wdno:$separator.
    }
  }
  MINUS {
    ?otherStatement a wdno:$separator.
    FILTER NOT EXISTS {
      ?statement a wdno:$separator.
    }
  }
EOF;
		return $filter;
	}

	/**
	 * @param Statement $statement
	 * @param PropertyId[] $separators
	 *
	 * @return CachedEntityIds
	 * @throws SparqlHelperException if the query times out or some other error occurs
	 */
	public function findEntitiesWithSameStatement(
		Statement $statement,
		array $separators
	) {
		$pid = $statement->getPropertyId()->getSerialization();
		$guid = $statement->getGuid();
		'@phan-var string $guid'; // statement must have a non-null GUID
		$guidForRdf = str_replace( '$', '-', $guid );

		$separatorFilters = array_map( [ $this, 'nestedSeparatorFilter' ], $separators );
		$finalSeparatorFilter = implode( "\n", $separatorFilters );

		$query = <<<EOF
SELECT DISTINCT ?otherEntity WHERE {
  BIND(wds:$guidForRdf AS ?statement)
  BIND(p:$pid AS ?p)
  BIND(ps:$pid AS ?ps)
  ?entity ?p ?statement.
  ?statement ?ps ?value.
  ?otherStatement ?ps ?value.
  ?otherEntity ?p ?otherStatement.
  FILTER(?otherEntity != ?entity)
  MINUS { ?otherStatement wikibase:rank wikibase:DeprecatedRank. }
  $finalSeparatorFilter
}
LIMIT 10
EOF;

		$result = $this->runQuery( $query );

		return $this->getOtherEntities( $result );
	}

	/**
	 * @param EntityId $entityId The entity ID on the containing entity
	 * @param PropertyValueSnak $snak
	 * @param string $type Context::TYPE_QUALIFIER or Context::TYPE_REFERENCE
	 * @param boolean $ignoreDeprecatedStatements Whether to ignore deprecated statements or not.
	 *
	 * @return CachedEntityIds
	 * @throws SparqlHelperException if the query times out or some other error occurs
	 */
	public function findEntitiesWithSameQualifierOrReference(
		EntityId $entityId,
		PropertyValueSnak $snak,
		$type,
		$ignoreDeprecatedStatements
	) {
		$eid = $entityId->getSerialization();
		$pid = $snak->getPropertyId()->getSerialization();
		$prefix = $type === Context::TYPE_QUALIFIER ? 'pq' : 'pr';
		$dataValue = $snak->getDataValue();
		$dataType = $this->propertyDataTypeLookup->getDataTypeIdForProperty(
			$snak->getPropertyId()
		);
		list( $value, $isFullValue ) = $this->getRdfLiteral( $dataType, $dataValue );
		if ( $isFullValue ) {
			$prefix .= 'v';
		}
		$path = $type === Context::TYPE_QUALIFIER ?
			"$prefix:$pid" :
			"prov:wasDerivedFrom/$prefix:$pid";

		$deprecatedFilter = '';
		if ( $ignoreDeprecatedStatements ) {
			$deprecatedFilter = <<< EOF
  MINUS { ?otherStatement wikibase:rank wikibase:DeprecatedRank. }
EOF;
		}

		$query = <<<EOF
SELECT DISTINCT ?otherEntity WHERE {
  BIND(wd:$eid AS ?entity)
  BIND($value AS ?value)
  ?entity ?p ?statement.
  ?statement $path ?value.
  ?otherStatement $path ?value.
  ?otherEntity ?otherP ?otherStatement.
  FILTER(?otherEntity != ?entity)
$deprecatedFilter
}
LIMIT 10
EOF;

		$result = $this->runQuery( $query );

		return $this->getOtherEntities( $result );
	}

	/**
	 * Return SPARQL code for a string literal with $text as content.
	 *
	 * @param string $text
	 *
	 * @return string
	 */
	private function stringLiteral( $text ) {
		return '"' . strtr( $text, [ '"' => '\\"', '\\' => '\\\\' ] ) . '"';
	}

	/**
	 * Extract and parse entity IDs from the ?otherEntity column of a SPARQL query result.
	 *
	 * @param CachedQueryResults $results
	 *
	 * @return CachedEntityIds
	 */
	private function getOtherEntities( CachedQueryResults $results ) {
		return new CachedEntityIds( array_map(
			function ( $resultBindings ) {
				$entityIRI = $resultBindings['otherEntity']['value'];
				foreach ( $this->entityPrefixes as $entityPrefix ) {
					$entityPrefixLength = strlen( $entityPrefix );
					if ( substr( $entityIRI, 0, $entityPrefixLength ) === $entityPrefix ) {
						try {
							return $this->entityIdParser->parse(
								substr( $entityIRI, $entityPrefixLength )
							);
						} catch ( EntityIdParsingException $e ) {
							// fall through
						}
					}

					return null;
				}

				return null;
			},
			$results->getArray()['results']['bindings']
		), $results->getMetadata() );
	}

	// phpcs:disable Generic.Metrics.CyclomaticComplexity,Squiz.WhiteSpace.FunctionSpacing
	/**
	 * Get an RDF literal or IRI with which the given data value can be matched in a query.
	 *
	 * @param string $dataType
	 * @param DataValue $dataValue
	 *
	 * @return array the literal or IRI as a string in SPARQL syntax,
	 * and a boolean indicating whether it refers to a full value node or not
	 */
	private function getRdfLiteral( $dataType, DataValue $dataValue ) {
		switch ( $dataType ) {
			case 'string':
			case 'external-id':
				return [ $this->stringLiteral( $dataValue->getValue() ), false ];
			case 'commonsMedia':
				$url = $this->rdfVocabulary->getMediaFileURI( $dataValue->getValue() );
				return [ '<' . $url . '>', false ];
			case 'geo-shape':
				$url = $this->rdfVocabulary->getGeoShapeURI( $dataValue->getValue() );
				return [ '<' . $url . '>', false ];
			case 'tabular-data':
				$url = $this->rdfVocabulary->getTabularDataURI( $dataValue->getValue() );
				return [ '<' . $url . '>', false ];
			case 'url':
				$url = $dataValue->getValue();
				if ( !preg_match( '/^[^<>"{}\\\\|^`\\x00-\\x20]*$/D', $url ) ) {
					// not a valid URL for SPARQL (see SPARQL spec, production 139 IRIREF)
					// such an URL should never reach us, so just throw
					throw new InvalidArgumentException( 'invalid URL: ' . $url );
				}
				return [ '<' . $url . '>', false ];
			case 'wikibase-item':
			case 'wikibase-property':
				/** @var EntityIdValue $dataValue */
				'@phan-var EntityIdValue $dataValue';
				return [ 'wd:' . $dataValue->getEntityId()->getSerialization(), false ];
			case 'monolingualtext':
				/** @var MonolingualTextValue $dataValue */
				'@phan-var MonolingualTextValue $dataValue';
				$lang = $dataValue->getLanguageCode();
				if ( !preg_match( '/^[a-zA-Z]+(-[a-zA-Z0-9]+)*$/D', $lang ) ) {
					// not a valid language tag for SPARQL (see SPARQL spec, production 145 LANGTAG)
					// such a language tag should never reach us, so just throw
					throw new InvalidArgumentException( 'invalid language tag: ' . $lang );
				}
				return [ $this->stringLiteral( $dataValue->getText() ) . '@' . $lang, false ];
			case 'globe-coordinate':
			case 'quantity':
			case 'time':
				// @phan-suppress-next-line PhanUndeclaredMethod
				return [ 'wdv:' . $dataValue->getHash(), true ];
			default:
				throw new InvalidArgumentException( 'unknown data type: ' . $dataType );
		}
	}
	// phpcs:enable

	/**
	 * @param string $text
	 * @param string $regex
	 *
	 * @return boolean
	 * @throws SparqlHelperException if the query times out or some other error occurs
	 * @throws ConstraintParameterException if the $regex is invalid
	 */
	public function matchesRegularExpression( $text, $regex ) {
		// caching wrapper around matchesRegularExpressionWithSparql

		$textHash = hash( 'sha256', $text );
		$cacheKey = $this->cache->makeKey(
			'WikibaseQualityConstraints', // extension
			'regex', // action
			'WDQS-Java', // regex flavor
			hash( 'sha256', $regex )
		);

		$cacheMapArray = $this->cache->getWithSetCallback(
			$cacheKey,
			WANObjectCache::TTL_DAY,
			function ( $cacheMapArray ) use ( $text, $regex, $textHash ) {
				// Initialize the cache map if not set
				if ( $cacheMapArray === false ) {
					$key = 'wikibase.quality.constraints.regex.cache.refresh.init';
					$this->dataFactory->increment( $key );
					return [];
				}

				$key = 'wikibase.quality.constraints.regex.cache.refresh';
				$this->dataFactory->increment( $key );
				$cacheMap = MapCacheLRU::newFromArray( $cacheMapArray, $this->cacheMapSize );
				if ( $cacheMap->has( $textHash ) ) {
					$key = 'wikibase.quality.constraints.regex.cache.refresh.hit';
					$this->dataFactory->increment( $key );
					$cacheMap->get( $textHash ); // ping cache
				} else {
					$key = 'wikibase.quality.constraints.regex.cache.refresh.miss';
					$this->dataFactory->increment( $key );
					try {
						$matches = $this->matchesRegularExpressionWithSparql( $text, $regex );
					} catch ( ConstraintParameterException $e ) {
						$matches = $this->serializeConstraintParameterException( $e );
					} catch ( SparqlHelperException $e ) {
						// don’t cache this
						return $cacheMap->toArray();
					}
					$cacheMap->set(
						$textHash,
						$matches,
						3 / 8
					);
				}

				return $cacheMap->toArray();
			},
			[
				// Once map is > 1 sec old, consider refreshing
				'ageNew' => 1,
				// Update 5 seconds after "ageNew" given a 1 query/sec cache check rate
				'hotTTR' => 5,
				// avoid querying cache servers multiple times in a request
				// (e. g. when checking format of a reference URL used multiple times on an entity)
				'pcTTL' => WANObjectCache::TTL_PROC_LONG,
			]
		);

		if ( isset( $cacheMapArray[$textHash] ) ) {
			$key = 'wikibase.quality.constraints.regex.cache.hit';
			$this->dataFactory->increment( $key );
			$matches = $cacheMapArray[$textHash];
			if ( is_bool( $matches ) ) {
				return $matches;
			} elseif ( is_array( $matches ) &&
				$matches['type'] == ConstraintParameterException::class ) {
				throw $this->deserializeConstraintParameterException( $matches );
			} else {
				throw new UnexpectedValueException(
					'Value of unknown type in object cache (' .
					'cache key: ' . $cacheKey . ', ' .
					'cache map key: ' . $textHash . ', ' .
					'value type: ' . gettype( $matches ) . ')'
				);
			}
		} else {
			$key = 'wikibase.quality.constraints.regex.cache.miss';
			$this->dataFactory->increment( $key );
			return $this->matchesRegularExpressionWithSparql( $text, $regex );
		}
	}

	private function serializeConstraintParameterException( ConstraintParameterException $cpe ) {
		return [
			'type' => ConstraintParameterException::class,
			'violationMessage' => $this->violationMessageSerializer->serialize( $cpe->getViolationMessage() ),
		];
	}

	private function deserializeConstraintParameterException( array $serialization ) {
		$message = $this->violationMessageDeserializer->deserialize(
			$serialization['violationMessage']
		);
		return new ConstraintParameterException( $message );
	}

	/**
	 * This function is only public for testing purposes;
	 * use matchesRegularExpression, which is equivalent but caches results.
	 *
	 * @param string $text
	 * @param string $regex
	 *
	 * @return boolean
	 * @throws SparqlHelperException if the query times out or some other error occurs
	 * @throws ConstraintParameterException if the $regex is invalid
	 */
	public function matchesRegularExpressionWithSparql( $text, $regex ) {
		$textStringLiteral = $this->stringLiteral( $text );
		$regexStringLiteral = $this->stringLiteral( '^(?:' . $regex . ')$' );

		$query = <<<EOF
SELECT (REGEX($textStringLiteral, $regexStringLiteral) AS ?matches) {}
EOF;

		$result = $this->runQuery( $query, false );

		$vars = $result->getArray()['results']['bindings'][0];
		if ( array_key_exists( 'matches', $vars ) ) {
			// true or false ⇒ regex okay, text matches or not
			return $vars['matches']['value'] === 'true';
		} else {
			// empty result: regex broken
			throw new ConstraintParameterException(
				( new ViolationMessage( 'wbqc-violation-message-parameter-regex' ) )
					->withInlineCode( $regex, Role::CONSTRAINT_PARAMETER_VALUE )
			);
		}
	}

	/**
	 * Check whether the text content of an error response indicates a query timeout.
	 *
	 * @param string $responseContent
	 *
	 * @return boolean
	 */
	public function isTimeout( $responseContent ) {
		$timeoutRegex = implode( '|', array_map(
			static function ( $fqn ) {
				return preg_quote( $fqn, '/' );
			},
			$this->timeoutExceptionClasses
		) );
		return (bool)preg_match( '/' . $timeoutRegex . '/', $responseContent );
	}

	/**
	 * Return the max-age of a cached response,
	 * or a boolean indicating whether the response was cached or not.
	 *
	 * @param array $responseHeaders see MWHttpRequest::getResponseHeaders()
	 *
	 * @return int|boolean the max-age (in seconds)
	 * or a plain boolean if no max-age can be determined
	 */
	public function getCacheMaxAge( $responseHeaders ) {
		if (
			array_key_exists( 'x-cache-status', $responseHeaders ) &&
			preg_match( '/^hit(?:-.*)?$/', $responseHeaders['x-cache-status'][0] )
		) {
			$maxage = [];
			if (
				array_key_exists( 'cache-control', $responseHeaders ) &&
				preg_match( '/\bmax-age=(\d+)\b/', $responseHeaders['cache-control'][0], $maxage )
			) {
				return intval( $maxage[1] );
			} else {
				return true;
			}
		} else {
			return false;
		}
	}

	/**
	 * Get the delay date of a 429 headered response, which is caused by
	 * throttling of to many SPARQL-Requests. The header-format is defined
	 * in RFC7231 see: https://tools.ietf.org/html/rfc7231#section-7.1.3
	 *
	 * @param MWHttpRequest $request
	 *
	 * @return int|ConvertibleTimestamp
	 * or SparlHelper::NO_RETRY_AFTER if there is no Retry-After header
	 * or SparlHelper::EMPTY_RETRY_AFTER if there is an empty Retry-After
	 * or SparlHelper::INVALID_RETRY_AFTER if there is something wrong with the format
	 */
	public function getThrottling( MWHttpRequest $request ) {
		$retryAfterValue = $request->getResponseHeader( 'Retry-After' );
		if ( $retryAfterValue === null ) {
			return self::NO_RETRY_AFTER;
		}

		$trimmedRetryAfterValue = trim( $retryAfterValue );
		if ( empty( $trimmedRetryAfterValue ) ) {
			return self::EMPTY_RETRY_AFTER;
		}

		if ( is_numeric( $trimmedRetryAfterValue ) ) {
			$delaySeconds = (int)$trimmedRetryAfterValue;
			if ( $delaySeconds >= 0 ) {
				return $this->getTimestampInFuture( new DateInterval( 'PT' . $delaySeconds . 'S' ) );
			}
		} else {
			$return = strtotime( $trimmedRetryAfterValue );
			if ( !empty( $return ) ) {
				return new ConvertibleTimestamp( $return );
			}
		}
		return self::INVALID_RETRY_AFTER;
	}

	private function getTimestampInFuture( DateInterval $delta ) {
		$now = new ConvertibleTimestamp();
		return new ConvertibleTimestamp( $now->timestamp->add( $delta ) );
	}

	/**
	 * Runs a query against the configured endpoint and returns the results.
	 * TODO: See if Sparql Client in core can be used instead of rolling our own
	 *
	 * @param string $query The query, unencoded (plain string).
	 * @param bool $needsPrefixes Whether the query requires prefixes or they can be omitted.
	 *
	 * @return CachedQueryResults
	 *
	 * @throws SparqlHelperException if the query times out or some other error occurs
	 */
	public function runQuery( $query, $needsPrefixes = true ) {

		if ( $this->throttlingLock->isLocked( self::EXPIRY_LOCK_ID ) ) {
			$this->dataFactory->increment( 'wikibase.quality.constraints.sparql.throttling' );
			throw new TooManySparqlRequestsException();
		}

		if ( $this->sparqlHasWikibaseSupport ) {
			$needsPrefixes = false;
		}

		if ( $needsPrefixes ) {
			$query = $this->prefixes . $query;
		}
		$query = "#wbqc\n" . $query;

		$url = $this->endpoint . '?' . http_build_query(
			[
				'query' => $query,
				'format' => 'json',
				'maxQueryTimeMillis' => $this->maxQueryTimeMillis,
			],
			'', ini_get( 'arg_separator.output' ),
			// encode spaces with %20, not +
			PHP_QUERY_RFC3986
		);

		$options = [
			'method' => 'GET',
			'timeout' => (int)round( ( $this->maxQueryTimeMillis + 1000 ) / 1000 ),
			'connectTimeout' => 'default',
			'userAgent' => $this->defaultUserAgent,
		];
		$request = $this->requestFactory->create( $url, $options, __METHOD__ );
		$startTime = microtime( true );
		$requestStatus = $request->execute();
		$endTime = microtime( true );
		$this->dataFactory->timing(
			'wikibase.quality.constraints.sparql.timing',
			( $endTime - $startTime ) * 1000
		);

		$this->guardAgainstTooManyRequestsError( $request );

		$maxAge = $this->getCacheMaxAge( $request->getResponseHeaders() );
		if ( $maxAge ) {
			$this->dataFactory->increment( 'wikibase.quality.constraints.sparql.cached' );
		}

		if ( $requestStatus->isOK() ) {
			$json = $request->getContent();
			$jsonStatus = FormatJson::parse( $json, FormatJson::FORCE_ASSOC );
			if ( $jsonStatus->isOK() ) {
				return new CachedQueryResults(
					$jsonStatus->getValue(),
					Metadata::ofCachingMetadata(
						$maxAge ?
							CachingMetadata::ofMaximumAgeInSeconds( $maxAge ) :
							CachingMetadata::fresh()
					)
				);
			} else {
				$jsonErrorCode = $jsonStatus->getErrors()[0]['message'];
				$this->dataFactory->increment(
					"wikibase.quality.constraints.sparql.error.json.$jsonErrorCode"
				);
				// fall through to general error handling
			}
		} else {
			$this->dataFactory->increment(
				"wikibase.quality.constraints.sparql.error.http.{$request->getStatus()}"
			);
			// fall through to general error handling
		}

		$this->dataFactory->increment( 'wikibase.quality.constraints.sparql.error' );

		if ( $this->isTimeout( $request->getContent() ) ) {
			$this->dataFactory->increment(
				'wikibase.quality.constraints.sparql.error.timeout'
			);
		}

		throw new SparqlHelperException();
	}

	/**
	 * Handle a potential “too many requests” error.
	 *
	 * @param MWHttpRequest $request
	 * @throws TooManySparqlRequestsException
	 */
	private function guardAgainstTooManyRequestsError( MWHttpRequest $request ): void {
		if ( $request->getStatus() !== self::HTTP_TOO_MANY_REQUESTS ) {
			return;
		}

		$fallbackBlockDuration = $this->sparqlThrottlingFallbackDuration;

		if ( $fallbackBlockDuration < 0 ) {
			throw new InvalidArgumentException( 'Fallback duration must be positive int but is: ' .
				$fallbackBlockDuration );
		}

		$this->dataFactory->increment( 'wikibase.quality.constraints.sparql.throttling' );
		$throttlingUntil = $this->getThrottling( $request );
		if ( !( $throttlingUntil instanceof ConvertibleTimestamp ) ) {
			$this->loggingHelper->logSparqlHelperTooManyRequestsRetryAfterInvalid( $request );
			$this->throttlingLock->lock(
				self::EXPIRY_LOCK_ID,
				$this->getTimestampInFuture( new DateInterval( 'PT' . $fallbackBlockDuration . 'S' ) )
			);
		} else {
			$this->loggingHelper->logSparqlHelperTooManyRequestsRetryAfterPresent( $throttlingUntil, $request );
			$this->throttlingLock->lock( self::EXPIRY_LOCK_ID, $throttlingUntil );
		}
		throw new TooManySparqlRequestsException();
	}

}
