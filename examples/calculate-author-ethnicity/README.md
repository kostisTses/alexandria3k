# Populating Crossref by author ethnicity

Contains a [Makefile](Makefile) that populates a database with all works associated with authors whose name matches a given ethnicity. 
Also creates a [Script](calculate-ethincity.sh) to make the interface more usable.

#### Steps: 
- Populates only id, given, family, work_id of the work_authors table.

- Classifies all authors based on given and family using the link-author-ethnicities process.

- Populates the rest of the tables only on the authors of the selected ethnicity

#### Usage:

```
./calculate-ethnicity.sh <ethnicty>
```

