// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/DressedLeptons.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Projections/PromptFinalState.hh"
#include "Rivet/Projections/MissingMomentum.hh"
#include <vector>
#include <cmath>

namespace Rivet {
  // Define classes and constructor:
  class pass_mono :

  public Analysis {
  public:

    /// Constructor
    pass_mono(const string name="pass_mono",
                        const string ref_data="pass_mono") : Analysis(name) {
      setRefDataName(ref_data);
    }
    /// Initialize
    void init() {
        
      // Get options from the new option system
      _mode = 0;
      if ( getOption("LMODE") == "NU" ) _mode = 0; // using Z -> nunu channel by default

      // Work in detector range eta=+- 4.9. (abolute value pseudorapidity (abseta) < 4.9)
      // Prompt photons. Declare final state photons
      PromptFinalState photon_fs(Cuts::abspid == PID::PHOTON && Cuts::abseta < 4.9);
      // Prompt electrons. Declare final state electrons.
      PromptFinalState el_fs(Cuts::abseta < 4.9 && Cuts::abspid == PID::ELECTRON);
      // Prompt muons. Declare final state muons.
      PromptFinalState mu_fs(Cuts::abseta < 4.9 && Cuts::abspid == PID::MUON);

      // Dressed leptons. Leptonic cut. Select leptons with pT> 7 GeV and detector range eta=+- 2.5
      Cut lep_cuts = Cuts::pT > 7*GeV && Cuts::abseta < 2.5;
      DressedLeptons dressed_leps(photon_fs, (_mode == 2 ? el_fs : mu_fs), 0.1, lep_cuts);
      declare(dressed_leps, "DressedLeptons");

      // In-acceptance leptons for lepton veto
      PromptFinalState veto_lep_fs(Cuts::abseta < 4.9 && (Cuts::abspid == PID::ELECTRON || Cuts::abspid == PID::MUON));
      veto_lep_fs.acceptTauDecays();
      veto_lep_fs.acceptMuonDecays();
      DressedLeptons veto_lep(photon_fs, veto_lep_fs, 0.1, lep_cuts);
      declare(veto_lep, "VetoLeptons");

      // MET
      VetoedFinalState met_fs(Cuts::abseta > 2.5 && Cuts::abspid == PID::MUON); // veto out-of-acceptance muons
      if (_mode) met_fs.addVetoOnThisFinalState(dressed_leps);
      declare(MissingMomentum(met_fs), "MET");

      // Jet collection
      FastJets jets(FinalState(Cuts::abseta < 4.9), FastJets::ANTIKT, 0.4, JetAlg::NO_MUONS, JetAlg::NO_INVISIBLES);
      declare(jets, "Jets");
        MSG_INFO("data: event,weight,njets,met,jpt1,jeta1,jpt2,jpt3,mjj,dphi,pass_vbf,pass_mono");
    }
      
    bool isBetweenJets(const Jet& probe, const Jet& boundary1, const Jet& boundary2) {
      const double y_p = probe.rapidity();
      const double y_b1 = boundary1.rapidity();
      const double y_b2 = boundary2.rapidity();
      const double y_min = std::min(y_b1, y_b2);
      const double y_max = std::max(y_b1, y_b2);
      return (y_p > y_min && y_p < y_max);
    }

    int centralJetVeto(Jets& jets) {
      if (jets.size() < 2) return 0;
      const Jet bj1 = jets.at(0);
      const Jet bj2 = jets.at(1);

      // Start loop at the 3rd hardest pT jet
      int n_between = 0;
      for (size_t i = 2; i < jets.size(); ++i) {
        const Jet j = jets.at(i);
        if (isBetweenJets(j, bj1, bj2) && j.pT() > 25*GeV)  ++n_between;
      }
      return n_between;
    }

      double sum_weight=0.0;
      double sur_events=0.0;
      
    /// Perform the per-event analysis
    void analyze(const Rivet::Event& event) {
    
    std::cout << "Starting event " << numEvents() << std::endl ;
    const double weight = event.weight(); // SoW counter
        
    sum_weight += weight;

      // Require 0 (Znunu) or 2 (Zll) dressed leptons
      bool isZll = bool(_mode);
      const vector<DressedLepton> &vetoLeptons = applyProjection<DressedLeptons>(event, "VetoLeptons").dressedLeptons();
      const vector<DressedLepton> &all_leps = applyProjection<DressedLeptons>(event, "DressedLeptons").dressedLeptons();
        if (!isZll && vetoLeptons.size())    vetoEvent;
        if ( isZll && all_leps.size() != 2)  vetoEvent;
        //isZll = 0 for all events. (Should be in mode 0)
        
      // Get jets:
        Jets jets = applyProjection<FastJets>(event, "Jets").jetsByPt(Cuts::pT > 25*GeV && Cuts::absrap < 4.4);
        
      const size_t njets = jets.size();
        
        FourMomentum PtMiss;
        for (unsigned i=0; i<njets; i++) {
            PtMiss+= jets[i].momentum();
        }
        double etmiss=PtMiss.pT();
        
        if (!njets)  vetoEvent;
      const int njets_gap = centralJetVeto(jets);

        
      double jpt1 = jets[0].pT(); // transverse momentum for 1st jet.
      double jeta1 = jets[0].eta(); // preseudorapidity for 1st jet.
      double mjj = 0., jpt2 = 0., jpt3=0., dphijj = 0.;
        
        // Need algorithm to tell apart jets. (Exclude soft jets).
        
      if (njets >= 2) {
        mjj = (jets[0].momentum() + jets[1].momentum()).mass();
        jpt2 = jets[1].pT();
        dphijj = deltaPhi(jets[0], jets[1]);
      }
      if (njets >=3) {jpt3 = jets[2].pT();}

      // MET
      Vector3 met_vec = apply<MissingMomentum>(event, "MET").vectorMPT();
      double met = met_vec.mod();

      // Cut on deltaPhi between MET and first 4 jets, but only if jet pT > 30 GeV
      bool dphi_fail = false;
      for (size_t i = 0; i < jets.size() && i < 4; ++i) {
        dphi_fail |= (deltaPhi(jets[i], met_vec) < 0.4 && jets[i].pT() > 30*GeV);
      }

     // Cuts & Event Selection:
      const bool pass_met_dphi = met > 200*GeV && !dphi_fail;
      const bool pass_vbf = pass_met_dphi && mjj > 200*GeV && jpt1 > 80*GeV && jpt2 > 50*GeV && njets >= 2 && !njets_gap;
      const bool pass_mono = pass_met_dphi && jpt1 > 120*GeV && fabs(jeta1) < 2.4;
    
        if (pass_mono) {
            sur_events+=1;
        }

    if(pass_mono)
    // Or alternatively, if pass_vbf, depending on what region of phase space we are considering.
    // Print out a few variables.
    {MSG_INFO("data: " << numEvents() << ", " << weight << ", " << njets << ", "<< met << ", " << jpt1 << ", " << jeta1 << ", " << jpt2 << ", " << mjj << ", " << dphijj << ", " << pass_vbf << ", " << pass_mono);}
    }
        
    /// Normalise, scale and otherwise manipulate histograms here
      void finalize() {
          std::cout << "Sum of weights: " << sum_weight << std::endl;
          std::cout << "Events surviving: " << sur_events << std::endl;
      const double sf(crossSection() / femtobarn / sumOfWeights());
          
        MSG_INFO(sf);
      }

  protected:
    // Analysis-mode switch
    size_t _mode;
  };
  // Hooks for the plugin system
DECLARE_RIVET_PLUGIN(pass_mono);
}
