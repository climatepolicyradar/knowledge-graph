# Wikibase Quality Constraints
[![Scrutinizer Code Quality][scrutinizer-badge]][scrutinizer]

Extension to Wikibase Repository that performs constraint checks.

[scrutinizer-badge]: https://scrutinizer-ci.com/g/wikimedia/mediawiki-extensions-WikibaseQualityConstraints/badges/quality-score.png?b=master
[scrutinizer]: https://scrutinizer-ci.com/g/wikimedia/mediawiki-extensions-WikibaseQualityConstraints/?branch=master

## Installation

* Clone `WikibaseQualityConstraints` inside the `extensions/` directory of your MediaWiki installations.

  ``` sh
  cd .../extensions/
  git clone https://gerrit.wikimedia.org/r/mediawiki/extensions/WikibaseQualityConstraints
  ```

* Install dependencies.
  The simplest way is to set up [composer-merge-plugin](https://www.mediawiki.org/wiki/Composer#Using_composer-merge-plugin)
  and then run `composer install` in the MediaWiki base directory;
  alternatively, you can run `composer install` inside the `WikibaseQualityConstraints` directory.

* Load the extension.

  ```php
  wfLoadExtension( 'WikibaseQualityConstraints' );
  ```

* Run `php maintenance/update.php --quick`.

* Configure the extension.
  You can find the configuration settings with documentation in `extension.json`.
  (Note that the variable name for `LocalSettings.php` always begins with `wg`,
  e. g. `$wgWBQualityConstraintsClassId` for the `WBQualityConstraintsClassId` setting.)

  * Specify the entity IDs of entities that are used to define constraints.
    See the “Data import” section for an automatic way to do this.

  * If you have a SPARQL endpoint, configure it in `WBQualityConstraintsSparqlEndpoint`.

  * Alternatively, to check “format” constraints without running a full SPARQL server,
    you can use the [minisparql] server.

* Run `php maintenance/run.php WikibaseQualityConstraints:ImportConstraintStatements.php`.

[minisparql]: https://github.com/lucaswerkmeister/minisparql

### Client-side script

The extension includes a client-side script that checks constraints on entity pages and displays violations on statements.
It is loaded by default for all logged-in users;
anonymous users can load it on a page by entering the following code into the web console:

```js
mw.loader.load( 'wikibase.quality.constraints.gadget' );
```

### Data import

For local development, it’s easiest to import some data from Wikidata.
You can use the `ImportConstraintEntities` maintenance script script to do this;
it will import all the required entities from Wikidata that don’t exist in your local wiki yet
and then print a config snippet which you can append to your `LocalSettings.php`.

```sh
# working directory should be the MediaWiki installation folder, i.e. where LocalSettings.php is
php maintenance/run.php WikibaseQualityConstraints:ImportConstraintEntities.php | tee -a LocalSettings.php
```

(The new entities will not show up in your wiki’s recent changes until they have been processed in the job queue;
try running the `maintenance/runJobs.php` script if it doesn’t happen automatically.)

### Running the tests

#### PHP

There are two ways to run the tests of this extension:

- Using the included configuration file:

  ```sh
  # from the MediaWiki installation folder
  composer phpunit -- -c extensions/WikibaseQualityConstraints/phpunit.xml.dist
  ```

  This creates test coverage reports
  (in `tests/coverage/` and `build/logs/clover.xml`)
  and is therefore fairly slow.

- Without the configuration file:

  ```sh
  # from the MediaWiki installation folder
  composer phpunit:entrypoint extensions/WikibaseQualityConstraints/tests/phpunit/
  ```

  This runs the tests without coverage report
  and is therefore much faster.

#### Javascript

You can run the tests, combined with linting and few other tools for asserting code quality by

```sh
  # from this extension's folder
  npm install
  grunt test
```

### Adding a new constraint type

To add a new constraint type, the following steps are necessary:

* Define the constraint checker class.
  * It should be defined in a new file in `src/ConstraintCheck/Checker/`,
    named after the class name.
    It should be in the `WikibaseQuality\ConstraintReport\ConstraintCheck\Checker` namespace.
  * The class name should follow the constraint type name (in English), ending in “Checker”.
  * The class must implement the `ConstraintChecker` interface.
  * It should have at least the following class-level documentation comment:
    ```php
    /**
     * @author YOUR NAME HERE
     * @license GPL-2.0-or-later
     */
    ```
  * Any services you need (`Config`, `EntityLookup`, …) should be injected as constructor parameters.
  * If the constraint has parameters,
    add support for parsing them to `ConstraintParameterParser`
    (add a config setting for the associated property in `extension.json`
    and a method to parse the parameter in `ConstraintParameterParser`),
    and then add tests for them in `ConstraintParameterParserTest`.
    This should be done in a separate commit.
* Define new messages (at least a violation message for the constraint type).
  * Define the message in `i18n/en.json`.
    A violation message should have a key like `wbqc-violation-message-constraintType`.
  * Document the message in `i18n/qqq.json`.
    Use the same message key,
    and insert the documentation in the same location where you also added the message in `en.json`
    (that is, `en.json` and `qqq.json` should contain message keys in the same order).
* Add a configuration setting for the constraint type item ID.
  * Configuration settings are defined in `extension.json`,
    as members of the `config` object.
  * It should be added right after the current last `…ConstraintId` entry.
  * It should be named after the constraint type item’s English label,
    following the pattern `WBQualityConstraints…ConstraintId`.
  * The default value should be the item ID on Wikidata,
    so that no extra configuration is required for Wikidata
    and importing the constraint type item (see “Data import” section) works.
  * The first part of the description can be copied from similar settings,
    the rest should contain a short description of the constraint type.
  * The ID can always be public (`"public": true`).
* Register the new constraint type checker.
  * In `ConstraintCheckerServices.php`, add a constant like
    ```php
    public const …_CHECKER = 'WBQC_…Checker';
    ```
    at the end of the list of constants.
    The value should be `'WBQC_'` followed by the class name,
    and the constant name should be the class name converted to all caps separated by underscores.
  * Also in `ConstraintCheckerServices.php`, add a method like
    ```php
    /**
     * @param MediaWikiServices|null $services
     * @return ConstraintChecker
     */
    public static function get…Checker( MediaWikiServices $services = null ) {
    	return self::getService( $services, self::…_CHECKER );
    }
    ```
    at the end of the class.
  * In `ServiceWiring-ConstraintCheckers.php`, append a new function like
    ```php
    ConstraintCheckerServices::…_CHECKER => function( MediaWikiServices $services ) {
    	return new …Checker(
        	// injected services
        );
    },
    ```
    to the array of services.
  * In `ServiceWiring.php`, append a new entry like
    ```php
    $config->get( 'WBQualityConstraints…ConstraintId' )
    	=> ConstraintCheckerServices::get…Checker( $services ),
    ```
    to the `$checkerMap` array in the `DELEGATING_CONSTRAINT_CHECKER` function.
  * In `ServicesTest.php`, append a new entry like
    ```php
    [ …Checker::class ],
    ```
    to the array in `provideConstraintCheckerServiceClasses()`.
* Add tests for the new constraint checker.
  * The test class name should be the same as the checker class name,
    with an additional suffix `Test` (i. e., `…CheckerTest`).
  * The test class should be placed somewhere in `tests/phpunit/Checker/`,
    either in the most suitable subdirectory
    or directly in that directory if none of the subdirectories are suitable.
    (The division into subdirectories there is dubious anyways,
    and we may get rid of it in the future.)
  * It should have at least the following class-level documentation comment:
    ```php
    /**
     * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\…Checker
     *
     * @group WikibaseQualityConstraints
     *
     * @author YOUR NAME HERE
     * @license GPL-2.0-or-later
     */
    ```
  * It should have at least one test for compliance with a constraint,
    one test for a constraint violation,
    one test for behavior on a deprecated statement,
    and one test for the `checkConstraintParameters` method.
  * Use the `ResultAssertions` trait’s methods to check constraint check results.
  * Use the `NewItem` and `NewStatement` builders to construct test data.
  * If the checker uses a `Config`, use the `DefaultConfig` trait.
  * If the constraint has parameters,
    add methods for them to the `ConstraintParameters` trait and use it in the tests.
  * You can copy+paste a `getConstraintMock` function from one of the existing tests,
    adjusting the `getConstraintTypeItemId` mocked return value.
    (Hopefully we’ll improve this in the future.)
* Update the tests for `DelegatingConstraintChecker`.
    * In `DelegatingConstraintCheckerTest`,
      add an entry for your constraint type to the `$constraints` array in `addDBData()`.
    * The `constraint_guid` should be `P1$`,
      followed by a new UUID (e. g. `cat /proc/sys/kernel/random/uuid` or `journalctl --new-id128`).
    * The `pid` should be `1`. (Not `'1'`!)
    * The `constraint_type_qid` should be `$this->getConstraintTypeItemId( '…' )`,
      where `…` is just the `…` part of the `WBQualityConstraints…ConstraintId` `extension.json` config key.
    * The `constraint_parameters` should be a valid JSON serialization of constraint parameters.
      If the constraint type doesn’t have any parameters, you can pass `{}`,
      otherwise there should ideally be methods to create the parameters in the `ConstraintParameters` trait
      so that you can use `json_encode( $this->…Parameter( … ) )`
      (perhaps with `array_merge` if there are multiple parameters).
* Ask someone with grafana-admin access to update the “constraint types” panel
  in the [wikidata-quality board](https://grafana.wikimedia.org/dashboard/db/wikidata-quality)
  to add the new constraint type.

### Adding a new allowed entity type

* Start by adding an Item reference to `extension.json`. For example:
 	```
 	"WBQualityConstraintsMediaInfoId": {
        "value": "Q3898244",
        "description": "The item ID of the 'MediaInfo' item, which represents the 'mediainfo' entity type for 'allowed entity types' constraints.",
        "public": true
    }
 * The next step takes place inside `src/ConstraintCheck/Helper/ConstraintParameterParser.php`. The new allowed entity type
 should be added to the switch/case within the method `parseEntityTypeItem()`.
 Don't forget to add it to the default case as well!
 * To be able to test the newly added allowed entity type locally, please perform the steps described in section "Data import".
