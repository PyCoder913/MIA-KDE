import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from KDE import KDE_GPU


class TrueDistributionAttack:
    def __init__(self, real_distances, unseen_distances, partition_param=1.0, split_param=0.3, equal=False, random_state=42, device='cpu'):
        """
        Initialize the TrueDistributionAttack class.

        Parameters
        ----------
        real_distances (list or np.array): Array of member (real training) distances.
        unseen_distances (list or np.array): Array of non-member (unseen) distances.
        partition_param (float): Fraction of real records to keep for the attack dataset (default = 1).
        split_param (float): Proportion of attack dataset to allocate to the test split (default = 0.3).
        equal (bool): If True, make sure to keep the same number of members and non-members in the test set (may not respect split_param test size)
        random_state (int): Random state for reproducibility.
        device: 'cpu' or 'gpu'
        """
        self.partition_param = partition_param
        self.split_param = split_param
        self.real_distances = np.asarray(real_distances)
        self.unseen_distances = np.asarray(unseen_distances)
        self.random_state = random_state
        self.device = device

        # Subsample real distances randomly according to partition parameter
        if self.partition_param <= 0 or self.partition_param > 1:
            raise ValueError("Invalid partition_param, please enter a value in (0,1]")
        if self.partition_param < 1:
            n_real = int(len(self.real_distances) * self.partition_param)
            rng = np.random.default_rng(self.random_state)
            sampled_indices = rng.choice(len(real_distances), size=n_real, replace=False)
            self.real_distances = real_distances[sampled_indices]

        # Create labels
        self.members_y = np.ones(len(self.real_distances))
        self.nonmembers_y = np.zeros(len(self.unseen_distances))

        # Split into training and testing
        self._split_data(equal)

        # Placeholder for fitted KDEs
        self.kde_member = None
        self.kde_nonmember = None
        self.membership_labels_ = None
        self.membership_probabilities_ = None

    def _split_data(self, equal):
        """Split randomly into train/test sets.
        There is no guarantee that the test set will contain an equal number of members and non-members, if equal=False.

        Parameters
        ----------
        equal (bool): Keep an equal number of members and non-members in the test set.
        
        If equal = True, the split_param parameter will not be respected in the overall train-test split. 
        The smaller class size will be determined (member or non-member), and it will be split using split_param.
        The same number of records will be sampled randomly from the larger class, and the test set will be formed.
        The rest of the records will be used for training (KDE fitting).
        
        """
        if not equal:
            X = np.concatenate([self.real_distances, self.unseen_distances])
            y = np.concatenate([
                np.ones(len(self.real_distances)),  # members
                np.zeros(len(self.unseen_distances))  # non-members
            ])
    
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.split_param, random_state=self.random_state, stratify=y
            )
    
            # Get training member/non-member splits for KDE fitting
            self.members_X_train = self.X_train[self.y_train == 1]
            self.nonmembers_X_train = self.X_train[self.y_train == 0]

        else:
            if self.partition_param == 1:
                # Split member and non-member distances into train/test sets
                self.members_X_train, self.members_X_test, self.members_y_train, self.members_y_test = train_test_split(
                    self.real_distances, self.members_y, test_size=self.split_param, random_state=self.random_state)
                self.nonmembers_X_train, self.nonmembers_X_test, self.nonmembers_y_train, self.nonmembers_y_test = train_test_split(
                    self.unseen_distances, self.nonmembers_y, test_size=self.split_param, random_state=self.random_state)
        
                self.X_train = np.concatenate([self.members_X_train, self.nonmembers_X_train])
                self.y_train = np.concatenate([self.members_y_train, self.nonmembers_y_train])
                self.X_test = np.concatenate([self.members_X_test, self.nonmembers_X_test])
                self.y_test = np.concatenate([self.members_y_test, self.nonmembers_y_test])

            else:
                rng = np.random.default_rng(self.random_state)
                
                # Balance members and non-members to same size
                min_size = min(len(self.real_distances), len(self.unseen_distances))
    
                if len(self.real_distances) < len(self.unseen_distances):
                    # Split the member distances using split_param, and sample the same number of test records from unseen distances
                    self.members_X_train, self.members_X_test, self.members_y_train, self.members_y_test = train_test_split(
                        sampled_real_distances, self.members_y, test_size=self.split_param, random_state=self.random_state)
                    self.nonmembers_X_test = rng.choice(self.unseen_distances, size=len(self.members_X_test), replace=False)
    
                    # The remaining records in unseen must go to training set
                    self.nonmembers_X_train = self.unseen_distances[~np.isin(self.unseen_distances, self.nonmembers_X_test)]
                    self.nonmembers_y_train = np.zeros(len(self.nonmembers_X_train))
                    self.nonmembers_y_test = np.zeros(len(self.nonmembers_X_test))
    
                else:
                    # Split the non-member distances using split_param, and sample the same number of test records from member distances
                    self.nonmembers_X_train, self.nonmembers_X_test, self.nonmembers_y_train, self.nonmembers_y_test = train_test_split(
                        self.unseen_distances, self.nonmembers_y, test_size=self.split_param, random_state=self.random_state)
                    self.members_X_test = rng.choice(self.real_distances, size=len(self.nonmembers_X_test), replace=False)
    
                    # The remaining records in member distances must go to training set
                    self.members_X_train = self.real_distances[~np.isin(self.real_distances, self.members_X_test)]
                    self.members_y_train = np.ones(len(self.members_X_train))
                    self.members_y_test = np.ones(len(self.members_X_test))
    
                self.X_train = np.concatenate([self.members_X_train, self.nonmembers_X_train])
                self.y_train = np.concatenate([self.members_y_train, self.nonmembers_y_train])
                self.X_test = np.concatenate([self.members_X_test, self.nonmembers_X_test])
                self.y_test = np.concatenate([self.members_y_test, self.nonmembers_y_test])

    def fit_kdes(self, bandwidth=None, rule='scott', kernel='gaussian', **kde_kwargs):
        """Fit KDEs on training distances for members and non-members.
        If device = 'cpu', then uses sklearn's KDE, otherwise our KDE_GPU.
        
        Parameters
        ----------
            bandwidth: Bandwidth for KDE, float or None. If None, use rule to compute bandwidth.
            rule: None, 'scott', or 'silverman'.
            kernel: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'.
            **kde_kwargs: Additional keyword arguments for the sklearn KDE constructor, if device='cpu'.
        """
        
        if self.device == 'cpu':
            if bandwidth is None and rule is not None:
                self.kde_member = KernelDensity(bandwidth=rule, kernel=kernel, **kde_kwargs)
                self.kde_nonmember = KernelDensity(bandwidth=rule, kernel=kernel, **kde_kwargs)
            elif rule is None and bandwidth is not None:
                self.kde_member = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kde_kwargs)
                self.kde_nonmember = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kde_kwargs)
            else:
                if rule not in [None, 'scott', 'silverman']:
                    raise ValueError("Unknown rule: use None, 'scott', or 'silverman'!")
                if rule is None and bandwidth is None:
                    raise ValueError("Both rule and bandwidth cannot be None")
                else:
                    raise ValueError("Cannot compute bandwidth with given inputs. Either give a float for bandwidth, or a rule")
            self.kde_member.fit(self.members_X_train.reshape(-1,1))
            self.kde_nonmember.fit(self.nonmembers_X_train.reshape(-1,1))

        if self.device == 'gpu':
            self.kde_member = KDE_GPU(kernel=kernel, rule=rule, bandwidth=bandwidth)
            self.kde_member.fit(self.members_X_train)
            self.kde_nonmember = KDE_GPU(kernel=kernel, rule=rule, bandwidth=bandwidth)
            self.kde_nonmember.fit(self.nonmembers_X_train)

    def evaluate(self, n_cores=4, batch_size=1000, prob_threshold=0.5):
        """Evaluate KDEs on test distances, calculate probabilities, and compute metrics.

        Parameters
        ----------
            n_cores (int): Number of cores to use in parallel KDE evaluation if using CPU.
            batch_size (int): Batch size for KDE evaluation if using GPU.
            prob_threshold (float, 0 to 1): Probability threshold for membership classification (default = 0.5).

        Returns
        ----------
            Accuracy and F1 score for membership classification on the test set.

        Attributes
        ----------
            membership_labels_: Membership predictions on the test set.
            membership_probs_: Membership probabilities.
        
        """
        X_test_eval = self.X_test.copy()
        if self.transformation:
            # Transform the test distances using the same transformation used before fitting
            X_test_eval = self._transform(X_test_eval, self.transformation)
            
        if self.device == 'cpu':
            def compute_density(kde, points):
                """Compute density scores for given points using a pre-fitted sklearn KDE"""
                return np.exp(kde.score_samples(points.reshape(-1, 1)))
            
            # Get densities for members
            chunks = np.array_split(X_test_eval, n_cores)
            results = Parallel(n_jobs=n_cores)(
                delayed(compute_density)(self.kde_member, chunk)
                for chunk in chunks
            )
            
            dens_member = np.concatenate(results, axis=0)
            
            # Get densities for non-members
            chunks = np.array_split(X_test_eval, n_cores)
            results = Parallel(n_jobs=n_cores)(
                delayed(compute_density)(self.kde_nonmember, chunk)
                for chunk in chunks
            )
            
            dens_nonmember = np.concatenate(results, axis=0)

        if self.device == 'gpu':        
            dens_member = self.kde_member.evaluate(X_test_eval, batch_size=batch_size) # densities for members
            dens_nonmember = self.kde_nonmember.evaluate(X_test_eval, batch_size=batch_size) # densities for non-members

        self.probs_member = dens_member / (dens_member + dens_nonmember)
        self.predictions = (self.probs_member >= prob_threshold).astype(int)

        acc = accuracy_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        self.membership_labels_ = np.asarray(self.predictions)
        self.membership_probs_ = np.asarray(self.probs_member)

        return acc, f1
      
class RealisticAttack:
    def __init__(self, real_distances, unseen_distances, partition_param=1.0, split_param=0.3, equal=False, random_state=42, device='cpu', threshold_percentile=50):
        """
        Initialize the TrueDistributionAttack class.

        Parameters
        ----------
        real_distances (list or np.array): Array of member (real training) distances.
        unseen_distances (list or np.array): Array of non-member (unseen) distances.
        partition_param (float): Fraction of real records to keep for the attack dataset (default = 1).
        split_param (float): Proportion of attack dataset to allocate to the test split (default = 0.3).
        equal (bool): If True, make sure to keep the same number of members and non-members in the test set (may not respect split_param test size).
        random_state (int): Random state for reproducibility.
        device: 'cpu' or 'gpu'.
        threshold_percentile (float): Percentile distance threshold to consider for deciding supposed members and non-members (default = 50).
        """
        self.partition_param = partition_param
        self.split_param = split_param
        self.real_distances = np.asarray(real_distances)
        self.unseen_distances = np.asarray(unseen_distances)
        self.random_state = random_state
        self.device = device
        self.threshold_percentile = threshold_percentile

        # Subsample real distances randomly according to partition parameter
        if self.partition_param <= 0 or self.partition_param > 1:
            raise ValueError("Invalid partition_param, please enter a value in (0,1]")
        if self.partition_param < 1:
            n_real = int(len(self.real_distances) * self.partition_param)
            rng = np.random.default_rng(self.random_state)
            sampled_indices = rng.choice(len(real_distances), size=n_real, replace=False)
            self.real_distances = real_distances[sampled_indices]

        # Create labels
        self.members_y = np.ones(len(self.real_distances))
        self.nonmembers_y = np.zeros(len(self.unseen_distances))

        # Split into training and testing
        self._split_data(equal)

        # Placeholder for fitted KDEs
        self.kde_member = None
        self.kde_nonmember = None
        self.membership_labels_ = None
        self.membership_probabilities_ = None

    def _split_data(self, equal):
        """Split randomly into train/test sets. The test set is not touched. The supposed members and supposed non-members are formed
              using the distance threshold from the train set.
        There is no guarantee that the test set will contain an equal number of members and non-members, if equal=False.

        Parameters
        ----------
        equal (bool): Keep an equal number of members and non-members in the test set.
        
        If equal = True, the split_param parameter will not be respected in the overall train-test split. 
        The smaller class size will be determined (member or non-member), and it will be split using split_param.
        The same number of records will be sampled randomly from the larger class, and the test set will be formed.
        The rest of the records will be used for training (KDE fitting).
        
        """
        if not equal:
            X = np.concatenate([self.real_distances, self.unseen_distances])
            y = np.concatenate([
                np.ones(len(self.real_distances)),  # members
                np.zeros(len(self.unseen_distances))  # non-members
            ])
    
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.split_param, random_state=self.random_state, stratify=y
            )
    
            # Get training member/non-member splits for KDE fitting
            self.members_X_train = self.X_train[self.y_train == 1]
            self.nonmembers_X_train = self.X_train[self.y_train == 0]

        else:
            if self.partition_param == 1:
                # Split member and non-member distances into train/test sets
                self.members_X_train, self.members_X_test, self.members_y_train, self.members_y_test = train_test_split(
                    self.real_distances, self.members_y, test_size=self.split_param, random_state=self.random_state)
                self.nonmembers_X_train, self.nonmembers_X_test, self.nonmembers_y_train, self.nonmembers_y_test = train_test_split(
                    self.unseen_distances, self.nonmembers_y, test_size=self.split_param, random_state=self.random_state)
        
                self.X_train = np.concatenate([self.members_X_train, self.nonmembers_X_train])
                self.y_train = np.concatenate([self.members_y_train, self.nonmembers_y_train])
                self.X_test = np.concatenate([self.members_X_test, self.nonmembers_X_test])
                self.y_test = np.concatenate([self.members_y_test, self.nonmembers_y_test])

            else:
                rng = np.random.default_rng(self.random_state)
                
                # Balance members and non-members to same size
                min_size = min(len(self.real_distances), len(self.unseen_distances))
    
                if len(self.real_distances) < len(self.unseen_distances):
                    # Split the member distances using split_param, and sample the same number of test records from unseen distances
                    self.members_X_train, self.members_X_test, self.members_y_train, self.members_y_test = train_test_split(
                        sampled_real_distances, self.members_y, test_size=self.split_param, random_state=self.random_state)
                    self.nonmembers_X_test = rng.choice(self.unseen_distances, size=len(self.members_X_test), replace=False)
    
                    # The remaining records in unseen must go to training set
                    self.nonmembers_X_train = self.unseen_distances[~np.isin(self.unseen_distances, self.nonmembers_X_test)]
                    self.nonmembers_y_train = np.zeros(len(self.nonmembers_X_train))
                    self.nonmembers_y_test = np.zeros(len(self.nonmembers_X_test))
    
                else:
                    # Split the non-member distances using split_param, and sample the same number of test records from member distances
                    self.nonmembers_X_train, self.nonmembers_X_test, self.nonmembers_y_train, self.nonmembers_y_test = train_test_split(
                        self.unseen_distances, self.nonmembers_y, test_size=self.split_param, random_state=self.random_state)
                    self.members_X_test = rng.choice(self.real_distances, size=len(self.nonmembers_X_test), replace=False)
    
                    # The remaining records in member distances must go to training set
                    self.members_X_train = self.real_distances[~np.isin(self.real_distances, self.members_X_test)]
                    self.members_y_train = np.ones(len(self.members_X_train))
                    self.members_y_test = np.ones(len(self.members_X_test))
    
                self.X_train = np.concatenate([self.members_X_train, self.nonmembers_X_train])
                self.y_train = np.concatenate([self.members_y_train, self.nonmembers_y_train])
                self.X_test = np.concatenate([self.members_X_test, self.nonmembers_X_test])
                self.y_test = np.concatenate([self.members_y_test, self.nonmembers_y_test])

    def fit_kdes(self, bandwidth=None, rule='scott', kernel='gaussian', **kde_kwargs):
        """Fit KDEs on training distances for members and non-members.
        If device = 'cpu', then uses sklearn's KDE, otherwise our KDE_GPU.
        
        Parameters
        ----------
            bandwidth: Bandwidth for KDE, float or None. If None, use rule to compute bandwidth.
            rule: None, 'scott', or 'silverman'.
            kernel: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'.
            **kde_kwargs: Additional keyword arguments for the sklearn KDE constructor, if device='cpu'.
        """

        threshold = np.percentile(self.X_train, self.threshold_percentile)
    
        # Form supposed member distances and supposed non-member distances from the training set
        self.supposed_member_distances = self.X_train[self.X_train <= threshold]
        self.supposed_nonmember_distances = self.X_train[self.X_train > threshold]
        
        if self.device == 'cpu':
            if bandwidth is None and rule is not None:
                self.kde_member = KernelDensity(bandwidth=rule, kernel=kernel, **kde_kwargs)
                self.kde_nonmember = KernelDensity(bandwidth=rule, kernel=kernel, **kde_kwargs)
            elif rule is None and bandwidth is not None:
                self.kde_member = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kde_kwargs)
                self.kde_nonmember = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kde_kwargs)
            else:
                if rule not in [None, 'scott', 'silverman']:
                    raise ValueError("Unknown rule: use None, 'scott', or 'silverman'!")
                if rule is None and bandwidth is None:
                    raise ValueError("Both rule and bandwidth cannot be None")
                else:
                    raise ValueError("Cannot compute bandwidth with given inputs. Either give a float for bandwidth, or a rule")
            self.kde_member.fit(self.supposed_member_distances.reshape(-1,1))
            self.kde_nonmember.fit(self.supposed_nonmember_distances.reshape(-1,1))

        if self.device == 'gpu':
            #from KDE_GPU import KDE_GPU
            self.kde_member = KDE_GPU(kernel=kernel, rule=rule, bandwidth=bandwidth)
            self.kde_member.fit(self.supposed_member_distances)
            self.kde_nonmember = KDE_GPU(kernel=kernel, rule=rule, bandwidth=bandwidth)
            self.kde_nonmember.fit(self.supposed_nonmember_distances)

    def evaluate(self, n_cores=4, batch_size=1000, prob_threshold=0.5):
        """Evaluate KDEs on test distances, calculate probabilities, and compute metrics.

        Parameters
        ----------
            n_cores (int): Number of cores to use in parallel KDE evaluation if using CPU.
            batch_size (int): Batch size for KDE evaluation if using GPU.
            prob_threshold (float, 0 to 1): Probability threshold for membership classification (default = 0.5).

        Returns
        ----------
            Accuracy and F1 score for membership classification on the test set.

        Attributes
        ----------
            membership_labels_: Membership predictions on the test set.
            membership_probs_: Membership probabilities.
        
        """
        X_test_eval = self.X_test.copy()
            
        if self.device == 'cpu':
            def compute_density(kde, points):
                """Compute density scores for given points using a pre-fitted sklearn KDE"""
                return np.exp(kde.score_samples(points.reshape(-1, 1)))
            
            # Get densities for members
            chunks = np.array_split(X_test_eval, n_cores)
            results = Parallel(n_jobs=n_cores)(
                delayed(compute_density)(self.kde_member, chunk)
                for chunk in chunks
            )
            
            dens_member = np.concatenate(results, axis=0)
            
            # Get densities for non-members
            chunks = np.array_split(X_test_eval, n_cores)
            results = Parallel(n_jobs=n_cores)(
                delayed(compute_density)(self.kde_nonmember, chunk)
                for chunk in chunks
            )
            
            dens_nonmember = np.concatenate(results, axis=0)

        if self.device == 'gpu':        
            dens_member = self.kde_member.evaluate(X_test_eval, batch_size=batch_size) # densities for members
            dens_nonmember = self.kde_nonmember.evaluate(X_test_eval, batch_size=batch_size) # densities for non-members

        self.probs_member = dens_member / (dens_member + dens_nonmember)
        self.predictions = (self.probs_member >= prob_threshold).astype(int)

        acc = accuracy_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        self.membership_labels_ = np.asarray(self.predictions)
        self.membership_probs_ = np.asarray(self.probs_member)

        return acc, f1
