# Changelog

All notable changes to the BicameralAGI project will be documented in this file.

## [Unreleased]

### Todo-list

#### Complete
- Complete `bica_character.py`
- Complete `bica_memory.py`

#### Validate
- Validate `bica_logging.py`
- Validate `bica_destiny.py`
- Validate `bica_context.py`
- Validate `bica_cognition.py`

#### Implement
- Implement `bica_subconscious.py`

### Functioning Files
- `gpt_handler.py`
- `bica_utilities.py`
- `bica_safety.py`
- `bica_action_executor.py`
- `bica_main.py`
- `bica_profile.py`

## 2024-10-04

### Changed
- `profile.py` so that it creates a full behavior profile rather than just traits and base emotions
- `character.py` it now utilizes the profile before responding
- Reorganized the entire project

## 2024-10-03

### Changed
- `bica_profile.py` so that it now creates profile files off of the character inputs

### Notes
- There is a noticeable bug in the generated profile files that will be fixed.
- The profile files are not utilized yet, but will be soon

## 2024-10-02

### Added
- Added a new extract definition function in the new `bica_character.py` file
- Also added the ability for the user to define the character they chat with

### Changed
- `bica_action_executor.py` & `bica_context.py` - Fixed an issue with calling `gpt_handler.py`
- `bica_main.py` - Improved font color and made it easier to read
- Renamed `bica_orchestrator.py` to `bica_character.py` and updated the contents to be more like a character box
- Renamed `bica_affect.py` to `bica_profile.py` for clarity reasons

### Removed
- Temporarily removed some modules for now so that the main branch can be functional at all times

## 2024-09-25

### Added
- `CHANGELOG.md` - so we can track changes more easily

### Changed
- `bica_utilities.py` - Added a delete json function

## 2024-09-23

### Added
- `bica_subconscious.py` - laying out the wireframe for where the subconscious code will be placed

### Changed
- `gpt_handler.py` - simplified the code so that it is only targeting OpenAI models. Will eventually be switching to Open Source models or a custom model.

