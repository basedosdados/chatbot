DATASET_DETAILS_QUERY = """
query getDatasetDetails($id: ID!) {
    allDataset(id: $id, first: 1) {
        edges {
            node {
                id
                name
                description
                organizations {
                    edges {
                        node {
                            name
                        }
                    }
                }
                themes {
                    edges {
                        node {
                            name
                        }
                    }
                }
                tags {
                    edges {
                        node {
                            name
                        }
                    }
                }
                tables {
                    edges {
                        node {
                            id
                            name
                            description
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
                name
                description
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
                            description
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
            }
        }
    }
}
"""
