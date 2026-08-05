# GraphQL queries fetch the explicit model translation columns (`namePt`/..., `descriptionPt`/...)
# rather than the unqualified `name`/`description` accessors, so the caller can select the thread's
# language with a pt fallback. Column names stay as the real BigQuery identifiers and are never localized.

DATASET_DETAILS_QUERY = """
query getDatasetDetails($id: ID!) {
    allDataset(id: $id, first: 1) {
        edges {
            node {
                id
                namePt
                nameEn
                nameEs
                descriptionPt
                descriptionEn
                descriptionEs
                organizations {
                    edges {
                        node {
                            namePt
                            nameEn
                            nameEs
                        }
                    }
                }
                themes {
                    edges {
                        node {
                            namePt
                            nameEn
                            nameEs
                        }
                    }
                }
                tags {
                    edges {
                        node {
                            namePt
                            nameEn
                            nameEs
                        }
                    }
                }
                tables {
                    edges {
                        node {
                            id
                            namePt
                            nameEn
                            nameEs
                            descriptionPt
                            descriptionEn
                            descriptionEs
                            temporalCoverage
                            cloudTables {
                                edges {
                                    node {
                                        gcpProjectId
                                        gcpDatasetId
                                        gcpTableId
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""

TABLE_DETAILS_QUERY = """
query getTableDetails($id: ID!) {
    allTable(id: $id, first: 1){
        edges {
            node {
                id
                namePt
                nameEn
                nameEs
                descriptionPt
                descriptionEn
                descriptionEs
                temporalCoverage
                cloudTables {
                    edges {
                        node {
                            gcpProjectId
                            gcpDatasetId
                            gcpTableId
                        }
                    }
                }
                columns {
                    edges {
                        node {
                            id
                            name
                            descriptionPt
                            descriptionEn
                            descriptionEs
                            measurementUnit
                            coveredByDictionary
                            isPartition
                            bigqueryType {
                                name
                            }
                            directoryPrimaryKey {
                                table {
                                    id
                                    cloudTables {
                                        edges {
                                            node {
                                                gcpDatasetId
                                                gcpTableId
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                dataset {
                    id
                }
            }
        }
    }
}
"""
