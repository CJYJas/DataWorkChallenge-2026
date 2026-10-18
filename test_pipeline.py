"""Focused checks for leakage boundaries, forecast integration and objective equivalence."""
import unittest
import numpy as np
import pandas as pd
from pipeline import derive, forecast, fitted_trends, loss, DEFAULTS, OBJECTIVES, FLUIDS

class PipelineChecks(unittest.TestCase):
    def setUp(self):
        dates=pd.date_range('2000-01-01',periods=20,freq='QS')
        oil=np.linspace(1000,600,20);water=1200-oil
        self.hist=pd.DataFrame({'Date':dates,'Oil_Rate':oil,'Gas_Rate':oil*.3633,'Water_Rate':water})
    def test_equal_objective_matches_original_definition(self):
        y=self.hist[FLUIDS].to_numpy();q=np.stack([y*.9,y*1.05]);scales=np.array([1000,400,600])
        expected=np.mean(np.sqrt(np.mean((q-y)**2,axis=1))/scales,axis=1)
        np.testing.assert_allclose(loss(q,y,self.hist.Date,scales,OBJECTIVES['equal']),expected)
    def test_wor_constant_liquid_identity_and_origin(self):
        ds=pd.date_range(self.hist.Date.iloc[-1],periods=81,freq='QS')
        q,_=forecast(self.hist,ds,'wor_constant_liquid',DEFAULTS)
        np.testing.assert_allclose(q[0],self.hist[FLUIDS].iloc[-1])
        np.testing.assert_allclose(q[:,0]+q[:,2],1200)
        self.assertTrue((q>=0).all());self.assertTrue(np.isfinite(q).all())
        self.assertTrue((np.diff(q[:,0])<=1e-8).all())
    def test_all_direct_forms_finite(self):
        ds=pd.date_range(self.hist.Date.iloc[-1],periods=81,freq='QS')
        for name in ['exp_linear_water','arps_linear_water','wor_constant_liquid','wor_declining_liquid']:
            q,_=forecast(self.hist,ds,name,DEFAULTS)
            self.assertEqual(q.shape,(81,3));self.assertTrue(np.isfinite(q).all());self.assertTrue((q>=0).all())
    def test_future_mutation_does_not_change_fit(self):
        cutoff=self.hist.Date.iloc[12];alter=self.hist.copy();alter.loc[alter.Date>cutoff,FLUIDS]*=100
        self.assertEqual(fitted_trends(self.hist[self.hist.Date<=cutoff],DEFAULTS),fitted_trends(alter[alter.Date<=cutoff],DEFAULTS))
    def test_cumulative_units(self):
        d=pd.DataFrame({'Date':pd.to_datetime(['2000-01-01','2000-01-11']), 'Oil_Rate':[100.,100.],'Gas_Rate':[36.33,36.33],'Water_Rate':[0.,0.]})
        self.assertAlmostEqual(derive(d).Np_STB.iloc[-1],1000.)
    def test_dates_before_origin_rejected(self):
        with self.assertRaises(ValueError):forecast(self.hist,pd.to_datetime(['2000-01-01']),'wor_constant_liquid',DEFAULTS)

if __name__=='__main__':unittest.main()
