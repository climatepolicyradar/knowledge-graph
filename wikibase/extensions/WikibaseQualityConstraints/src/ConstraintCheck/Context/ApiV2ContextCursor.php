<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Context;

/**
 * Abstract superclass of all context cursors for the second version of the API output format.
 * (This is currently the only supported format.)
 *
 * This output format is modeled after the Wikibase entity JSON format,
 * where an object with the members 'hash' and a list of 'reports' can appear
 * wherever the entity JSON format contains snaks.
 *
 * That is, the container is keyed by entity ID and contains objects with a 'claims' member,
 * which holds an object that is keyed by property ID and contains lists of statement objects,
 * which have an 'id' member, a 'mainsnak' snak,
 * 'qualifiers' keyed by property ID holding a list of snaks,
 * and a list of 'references' each having a 'hash' and 'snaks'
 * which are keyed by property ID and then hold a list of snaks.
 *
 * @license GPL-2.0-or-later
 */
abstract class ApiV2ContextCursor implements ContextCursor {

	/**
	 * Returns the 'claims' subcontainer.
	 *
	 * @param array[] &$container
	 * @return array
	 */
	protected function &getClaimsArray( array &$container ) {
		$entityId = $this->getEntityId();

		if ( !array_key_exists( $entityId, $container ) ) {
			$container[$entityId] = [];
		}
		$entityContainer = &$container[$entityId];

		if ( !array_key_exists( 'claims', $entityContainer ) ) {
			$entityContainer['claims'] = [];
		}
		$claimsArray = &$entityContainer['claims'];

		return $claimsArray;
	}

	/**
	 * Returns the statement subcontainer.
	 *
	 * @param array[] &$container
	 * @return array
	 */
	protected function &getStatementArray( array &$container ) {
		$statementPropertyId = $this->getStatementPropertyId();
		$statementGuid = $this->getStatementGuid();

		$claimsContainer = &$this->getClaimsArray( $container );

		if ( !array_key_exists( $statementPropertyId, $claimsContainer ) ) {
			$claimsContainer[$statementPropertyId] = [];
		}
		$propertyContainer = &$claimsContainer[$statementPropertyId];

		foreach ( $propertyContainer as &$statement ) {
			if ( $statement['id'] === $statementGuid ) {
				$statementArray = &$statement;
				break;
			}
		}
		if ( !isset( $statementArray ) ) {
			$statementArray = [ 'id' => $statementGuid ];
			$propertyContainer[] = &$statementArray;
		}

		return $statementArray;
	}

	/**
	 * This method returns the array with the 'hash' and 'reports' member.
	 * It should locate the array in $container or,
	 * if the array doesn’t exist yet, create it and emplace it there.
	 *
	 * @param array[] &$container
	 * @return array
	 */
	abstract protected function &getMainArray( array &$container );

	/**
	 * @param ?array $result
	 * @param array[] &$container
	 */
	public function storeCheckResultInArray( ?array $result, array &$container ) {
		$mainArray = &$this->getMainArray( $container );
		if ( !array_key_exists( 'results', $mainArray ) ) {
			$mainArray['results'] = [];
		}

		if ( $result !== null ) {
			$mainArray['results'][] = $result;
		}
	}

}
