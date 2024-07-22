from scipy.sparse import csr_matrix
import numpy as np

class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        col_nums = matrix.sum(axis=0)
        norm_matrix = matrix.multiply(1 / col_nums)
        return csr_matrix(norm_matrix)

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        row_nums = matrix.sum(axis=1)
        norm_matrix = matrix.multiply(1 / row_nums)
        return csr_matrix(norm_matrix)

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        # Calculate Term Frequency (TF) by normalizing along rows
        tf_matrix = Normalization.by_row(matrix)

        # Calculate Inverse Document Frequency (IDF) as a numpy array
        doc_freq = np.sum(matrix > 0, axis=0)
        idf_vector = np.log((matrix.shape[0]) / doc_freq)  # Add 1 to avoid division by zero

        # Calculate TF-IDF using element-wise multiplication
        tf_idf_matrix = tf_matrix.multiply(idf_vector)

        return csr_matrix(tf_idf_matrix)


    @staticmethod
    def bm_25(
        matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        # Constants
        tf_matrix = Normalization.by_row(matrix)
        doc_freq = np.sum(matrix > 0, axis=0)
        idf = np.log((matrix.shape[0]) / doc_freq)

        # Calculate document-specific delta for each document
        doc_lengths = matrix.sum(axis=1)  # Convert sparse matrix to 1D array
        avg_dl = doc_lengths.mean()
        delta_data = k1 * (((1 - b) + b * (doc_lengths / avg_dl)))

        #tf_prime = matrix.copy()

        # Efficiently compute TF' = (TF / delta)^(-1)
        tf_matrix = tf_matrix.multiply(1 / delta_data)
        tf_matrix = tf_matrix.power(-1)
        tf_matrix.data += 1
        tf_matrix = tf_matrix.power(-1)

        # Efficiently compute TF' = TF'^(-1) * (k1 + 1)
        tf_matrix.data *= (k1 + 1)

        # Element-wise multiplication with idf
        norm_matrix = tf_matrix.multiply(idf)

        return csr_matrix(norm_matrix)


