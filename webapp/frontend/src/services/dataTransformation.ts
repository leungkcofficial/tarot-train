/**
 * Data Transformation Service for TAROT CKD Risk Prediction
 * Converts frontend form data to 11 features x 10 timepoints format for API
 */

import { Demographics, LabValue, Comorbidity } from '../contexts/SessionContext';
import { formatDate } from './api';

// 11 required features as per features_config.md
export const REQUIRED_FEATURES = [
  'age_at_obs',
  'albumin',
  'uacr', 
  'bicarbonate',
  'cci_score_total',
  'creatinine',
  'gender',
  'hemoglobin',
  'ht',
  'observation_period',
  'phosphate'
] as const;

export type FeatureName = typeof REQUIRED_FEATURES[number];

// Maximum timepoints supported
export const MAX_TIMEPOINTS = 10;

// Feature matrix: 11 features x 10 timepoints
export interface FeatureMatrix {
  [key: string]: (number | null)[];
}

// Timepoint data structure
export interface TimePoint {
  date: Date;
  features: Partial<Record<FeatureName, number>>;
}

// API request format for temporal data
export interface TemporalPredictionRequest {
  feature_matrix: FeatureMatrix;
  timepoint_dates: string[];
  patient_info: {
    age_at_obs: number;
    gender: 'male' | 'female';
    observation_period: number;
  };
  session_id?: string;
}

/**
 * Main data transformation class
 */
export class DataTransformationService {
  
  /**
   * Transform frontend form data to 11x10 feature matrix
   */
  static transformToFeatureMatrix(
    demographics: Demographics,
    labValues: LabValue[],
    comorbidities: Comorbidity[]
  ): TemporalPredictionRequest {
    
    // Validate required demographics
    if (!demographics.age || !demographics.gender) {
      throw new Error('Demographics: age and gender are required');
    }

    // Calculate observation period from dates (as per user requirements)
    const observationPeriod = this.calculateObservationPeriod(demographics, labValues);

    // Create timepoints from all lab value dates with UACR/UPCR priority logic
    const timepoints = this.createTimepoints(labValues, demographics, comorbidities, observationPeriod);
    
    // Sort timepoints by date (most recent first)
    timepoints.sort((a, b) => b.date.getTime() - a.date.getTime());
    
    // Limit to MAX_TIMEPOINTS
    const limitedTimepoints = timepoints.slice(0, MAX_TIMEPOINTS);
    
    // Initialize feature matrix
    const featureMatrix: FeatureMatrix = {};
    REQUIRED_FEATURES.forEach(feature => {
      featureMatrix[feature] = new Array(MAX_TIMEPOINTS).fill(null);
    });
    
    // Fill feature matrix with timepoint data
    limitedTimepoints.forEach((timepoint, index) => {
      Object.entries(timepoint.features).forEach(([feature, value]) => {
        if (value !== undefined && value !== null) {
          featureMatrix[feature][index] = value;
        }
      });
    });
    
    // Forward fill missing values for static features
    this.forwardFillStaticFeatures(featureMatrix);
    
    // Get timepoint dates
    const timepointDates = limitedTimepoints.map(tp => formatDate(tp.date));
    
    // Pad with null dates if needed
    while (timepointDates.length < MAX_TIMEPOINTS) {
      timepointDates.push('');
    }
    
    return {
      feature_matrix: featureMatrix,
      timepoint_dates: timepointDates,
      patient_info: {
        age_at_obs: demographics.age,
        gender: demographics.gender as 'male' | 'female',
        observation_period: observationPeriod
      }
    };
  }
  
  /**
   * Create timepoints from lab values and static features
   */
  private static createTimepoints(
    labValues: LabValue[],
    demographics: Demographics,
    comorbidities: Comorbidity[],
    observationPeriod: number
  ): TimePoint[] {
    
    // Get all unique dates from lab values
    const uniqueDates = Array.from(new Set(
      labValues.map(lab => lab.date.toISOString().split('T')[0])
    )).map(dateStr => new Date(dateStr));
    
    // If no lab dates, create at least one timepoint with current date
    if (uniqueDates.length === 0) {
      uniqueDates.push(new Date());
    }
    
    // Create timepoints
    const timepoints: TimePoint[] = uniqueDates.map(date => {
      const timepoint: TimePoint = {
        date,
        features: {}
      };
      
      // Add static features (same for all timepoints)
      timepoint.features.age_at_obs = demographics.age;
      timepoint.features.gender = demographics.gender === 'male' ? 1 : 0;
      timepoint.features.observation_period = observationPeriod;
      timepoint.features.ht = this.getHypertensionStatus(comorbidities);
      timepoint.features.cci_score_total = this.calculateCharlsonScore(comorbidities);
      
      // Add lab values for this date
      const labsForDate = labValues.filter(lab => 
        lab.date.toISOString().split('T')[0] === date.toISOString().split('T')[0]
      );
      
      // Handle UACR/UPCR priority logic: UACR takes priority over UPCR at same timepoint
      const uacrLab = labsForDate.find(lab => lab.parameter === 'uacr');
      const upcrLab = labsForDate.find(lab => lab.parameter === 'upcr');
      
      // Process regular lab values (non-uacr/upcr)
      labsForDate.forEach(lab => {
        if (lab.parameter !== 'uacr' && lab.parameter !== 'upcr') {
          const convertedValue = this.convertLabValue(lab);
          if (convertedValue !== null) {
            switch (lab.parameter) {
              case 'creatinine':
                timepoint.features.creatinine = convertedValue;
                break;
              case 'hemoglobin':
                timepoint.features.hemoglobin = convertedValue;
                break;
              case 'phosphate':
                timepoint.features.phosphate = convertedValue;
                break;
              case 'bicarbonate':
                timepoint.features.bicarbonate = convertedValue;
                break;
              case 'albumin':
                timepoint.features.albumin = convertedValue;
                break;
            }
          }
        }
      });
      
      // Handle UACR/UPCR priority: UACR preferred, use UPCR only if UACR not available
      if (uacrLab && uacrLab.value !== '' && uacrLab.value !== 0) {
        // Use UACR (priority)
        const convertedValue = this.convertLabValue(uacrLab);
        if (convertedValue !== null) {
          timepoint.features.uacr = convertedValue;
        }
      } else if (upcrLab && upcrLab.value !== '' && upcrLab.value !== 0) {
        // Use UPCR (fallback) - convert to UACR-equivalent for feature matrix
        const convertedValue = this.convertLabValue(upcrLab);
        if (convertedValue !== null) {
          // Store as UACR feature (converted from UPCR)
          timepoint.features.uacr = this.convertUpcrToUacr(convertedValue);
        }
      }
      
      return timepoint;
    });
    
    return timepoints;
  }
  
  /**
   * Convert lab value to standard units as per features_config.md
   */
  private static convertLabValue(lab: LabValue): number | null {
    if (typeof lab.value !== 'number' || lab.value === 0) {
      return null;
    }
    
    const value = lab.value;
    
    switch (lab.parameter) {
      case 'creatinine':
        // Target: µmol/L
        if (lab.unit === 'mg/dL') {
          return value * 88.4; // mg/dL to µmol/L
        }
        return value; // Already in µmol/L
        
      case 'hemoglobin':
        // Target: g/dL
        if (lab.unit === 'g/L') {
          return value / 10; // g/L to g/dL
        }
        return value; // Already in g/dL
        
      case 'phosphate':
        // Target: mmol/L
        if (lab.unit === 'mg/dL') {
          return value * 0.323; // mg/dL to mmol/L
        }
        return value; // Already in mmol/L
        
      case 'bicarbonate':
        // Target: mmol/L (same as mEq/L)
        return value;
        
      case 'albumin':
        // Target: g/L
        if (lab.unit === 'g/dL') {
          return value * 10; // g/dL to g/L
        }
        return value; // Already in g/L
        
      case 'uacr':
        // Target: mg/g
        if (lab.unit === 'mg/mmol') {
          return value * 8.84; // mg/mmol to mg/g
        }
        return value; // Already in mg/g
        
      default:
        return value;
    }
  }
  
  /**
   * Get hypertension status (HT feature)
   */
  private static getHypertensionStatus(comorbidities: Comorbidity[]): number {
    const hypertension = comorbidities.find(c => c.condition === 'hypertension');
    return hypertension?.diagnosed ? 1 : 0;
  }
  
  /**
   * Calculate Charlson Comorbidity Index (excluding hypertension)
   */
  private static calculateCharlsonScore(comorbidities: Comorbidity[]): number {
    // Charlson weights (excluding hypertension)
    const charlsonWeights: Record<string, number> = {
      'myocardial_infarction': 1,
      'congestive_heart_failure': 1,
      'peripheral_vascular_disease': 1,
      'cerebrovascular_disease': 1,
      'diabetes_uncomplicated': 1,
      'diabetes_complicated': 2,
      'chronic_pulmonary_disease': 1,
      'connective_tissue_disease': 1,
      'peptic_ulcer_disease': 1,
      'liver_disease_mild': 1,
      'liver_disease_severe': 3,
      'malignancy': 2,
      'metastatic_cancer': 6,
      'dementia': 1,
      'hemiplegia': 2,
      'aids': 6
    };
    
    return comorbidities
      .filter(c => c.diagnosed && c.condition !== 'hypertension')
      .reduce((total, condition) => {
        const weight = charlsonWeights[condition.condition] || 0;
        return total + weight;
      }, 0);
  }
  
  /**
   * Forward fill static features that don't change over time
   */
  private static forwardFillStaticFeatures(featureMatrix: FeatureMatrix): void {
    const staticFeatures = ['age_at_obs', 'gender', 'observation_period', 'ht', 'cci_score_total'];
    
    staticFeatures.forEach(feature => {
      const values = featureMatrix[feature];
      let lastValue: number | null = null;
      
      for (let i = 0; i < values.length; i++) {
        if (values[i] !== null) {
          lastValue = values[i];
        } else if (lastValue !== null) {
          values[i] = lastValue;
        }
      }
    });
  }
  
  /**
   * Convert UPCR value to UACR equivalent
   * Based on established clinical conversion formulas
   */
  private static convertUpcrToUacr(upcrValue: number): number {
    // Simplified conversion: UACR ≈ UPCR * 0.6-0.7 (rough approximation)
    // In practice, this would use a more sophisticated conversion algorithm
    // For now, using a conservative conversion factor
    return upcrValue * 0.65;
  }
  
  /**
   * Calculate observation period: today - earliest(CKD diagnosis date, earliest creatinine date)
   * Only creatinine dates are used as per user requirements
   */
  private static calculateObservationPeriod(demographics: Demographics, labValues: LabValue[]): number {
    // Get earliest creatinine date
    const creatinineValues = labValues.filter(lab => lab.parameter === 'creatinine');
    
    if (creatinineValues.length === 0) {
      throw new Error('At least one creatinine value is required to calculate observation period');
    }
    
    // Find earliest creatinine date
    const earliestCreatinineDate = creatinineValues
      .map(lab => lab.date)
      .reduce((earliest, current) => current < earliest ? current : earliest);
    
    // Get earliest date between CKD diagnosis date and earliest creatinine date
    let earliestDate = earliestCreatinineDate;
    
    if (demographics.ckdDiagnosisDate && demographics.ckdDiagnosisDate < earliestCreatinineDate) {
      earliestDate = demographics.ckdDiagnosisDate;
    }
    
    // Calculate days from earliest date to today
    const today = new Date();
    const timeDiffMs = today.getTime() - earliestDate.getTime();
    const daysDiff = Math.floor(timeDiffMs / (1000 * 60 * 60 * 24));
    
    return Math.max(0, daysDiff); // Ensure non-negative
  }
  
  /**
   * Validate feature matrix completeness
   */
  static validateFeatureMatrix(request: TemporalPredictionRequest): string[] {
    const errors: string[] = [];
    const { feature_matrix } = request;
    
    // Check if all required features are present
    REQUIRED_FEATURES.forEach(feature => {
      if (!feature_matrix[feature]) {
        errors.push(`Missing required feature: ${feature}`);
      } else {
        // Check if feature has at least one non-null value
        const hasValues = feature_matrix[feature].some(value => value !== null);
        if (!hasValues) {
          errors.push(`Feature ${feature} has no valid values`);
        }
      }
    });
    
    // Check critical lab values
    const criticalLabFeatures = ['creatinine', 'hemoglobin', 'albumin', 'uacr'];
    criticalLabFeatures.forEach(feature => {
      if (feature_matrix[feature]) {
        const hasRecentValue = feature_matrix[feature][0] !== null; // Most recent timepoint
        if (!hasRecentValue) {
          errors.push(`Missing recent value for critical feature: ${feature}`);
        }
      }
    });
    
    return errors;
  }
  
  /**
   * Get feature matrix summary for debugging
   */
  static getFeatureMatrixSummary(featureMatrix: FeatureMatrix): Record<string, any> {
    const summary: Record<string, any> = {};
    
    Object.entries(featureMatrix).forEach(([feature, values]) => {
      const nonNullValues = values.filter(v => v !== null);
      summary[feature] = {
        total_values: nonNullValues.length,
        latest_value: values[0],
        value_range: nonNullValues.length > 0 ? {
          min: Math.min(...(nonNullValues as number[])),
          max: Math.max(...(nonNullValues as number[])),
          mean: (nonNullValues as number[]).reduce((a, b) => a + b, 0) / nonNullValues.length
        } : null
      };
    });
    
    return summary;
  }
}

export default DataTransformationService;