Feature Comparison

| Feature Type       | SimpleExtractor | AdvancedExtractor | ExpertExtractor | DeepExtractor |
|-------------------|----------------|-----------------|----------------|---------------|
| Food Distance      | ✓              | ✓               | ✓              | ✓             |
| Food Density       | ✗              | ✓               | ✓              | ✓             |
| Ghost Danger       | Basic          | Advanced        | Predictive     | Interactive   |
| Scared Ghosts      | ✗              | ✓               | Strategic      | ✓             |
| Capsule Strategy   | ✗              | ✓               | ✓              | ✓             |
| Path Safety        | ✗              | ✓               | ✓              | ✓             |
| Territory Control  | ✗              | ✗               | ✓              | ✓             |
| Escape Routes      | ✗              | ✗               | ✓              | ✓             |
| Interaction Terms  | ✗              | ✗               | ✗              | ✓             |


These extractors should significantly improve Pacman's performance on challenging

```
# Test AdvancedExtractor
python pacman.py -p ApproximateQAgent -a extractor=AdvancedExtractor -x 50 -n 60 -l mediumGrid

# Test ExpertExtractor  
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 100 -n 110 -l mediumClassic

# Test DeepExtractor (may need more training)
python pacman.py -p ApproximateQAgent -a extractor=DeepExtractor -x 200 -n 210 -l mediumClassic

# Large challenging layout
python pacman.py -p ApproximateQAgent -a extractor=ExpertExtractor -x 100 -n 110 -l openClassic
```
